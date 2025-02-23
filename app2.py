import json
import csv
import logging
import os
import re
from io import BytesIO
from collections import Counter

import pandas as pd
import streamlit as st

# pdfminer for PDF extraction
from pdfminer.high_level import extract_text as pdf_extract_text

# AzureOpenAI client
from openai import AzureOpenAI

# For phone number validation
import phonenumbers

# For fuzzy matching (deduplication)
from rapidfuzz import fuzz

# For Russian NLP with Natasha
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc

# For visualization
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# ---------------------------------------
# Configuration
# ---------------------------------------
AZURE_API_KEY = "9abc905da5104e8eb8d6ec3ceb27f767"  # Replace with your actual API key
AZURE_ENDPOINT = "https://aoai.apim.mitre.org/api-key"  # Replace with your actual endpoint
API_VERSION = "2023-03-15-preview"

# ---------------------------------------
# Logging Configuration
# ---------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------
# Natasha Initialization for Russian NER
# ---------------------------------------
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# ---------------------------------------
# Validation Functions
# ---------------------------------------
def is_valid_ip(ip):
    pattern = r"^(25[0-5]|2[0-4]\d|[01]?\d?\d)\." \
              r"(25[0-5]|2[0-4]\d|[01]?\d?\d)\." \
              r"(25[0-5]|2[0-4]\d|[01]?\d?\d)\." \
              r"(25[0-5]|2[0-4]\d|[01]?\d?\d)$"
    if re.match(pattern, ip.strip()):
        return True
    if ":" in ip.strip():  # Basic IPv6 check
        return True
    return False

def is_valid_email(email):
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email.strip()))

def is_valid_phone_number(phone_str):
    try:
        parsed = phonenumbers.parse(phone_str, None)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False

def is_valid_iban(iban):
    # Naive IBAN check (we're keeping it in case you need it later)
    iban = iban.replace(" ", "")
    return 15 <= len(iban) <= 34

# ---------------------------------------
# PDF Extraction & Cleaning
# ---------------------------------------
def extract_text_from_pdf(pdf_file_path):
    try:
        text = pdf_extract_text(pdf_file_path)
        logger.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_file_path}: {e}")
        return ""

def clean_text(text):
    if not text:
        return ""
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(cleaned_text.split())

def chunk_text(text, max_chars=2000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    logger.info(f"Divided text into {len(chunks)} chunks.")
    logger.debug(f"First chunk preview: {chunks[0][:200]}")
    return chunks

# ---------------------------------------
# Azure OpenAI Call
# ---------------------------------------
def call_azure_openai_api(prompt, openai_client, model_name="gpt35-turbo-16k"):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI that extracts and normalizes entities/relationships from text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    if response.choices:
        result = response.choices[0].message.content.strip()
        logger.debug(f"API raw response: {result}")
        return result
    else:
        logger.error("No response from Azure OpenAI.")
        return ""

# ---------------------------------------
# Russian NER with Natasha
# ---------------------------------------
def detect_russian_entities(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    russian_names = set()
    russian_companies = set()
    for span in doc.spans:
        span.normalize(morph_vocab)
        if span.type == 'PER':
            russian_names.add(span.text)
        elif span.type == 'ORG':
            russian_companies.add(span.text)
    logger.debug(f"Natasha detected {len(russian_names)} person names and {len(russian_companies)} organizations.")
    return russian_names, russian_companies

def combine_entities(llm_entities, text):
    rus_names, rus_companies = detect_russian_entities(text)
    llm_entities["person_names"] = list(set(llm_entities.get("person_names", [])) | rus_names)
    llm_entities["company_names"] = list(set(llm_entities.get("company_names", [])) | rus_companies)
    return llm_entities

# ---------------------------------------
# Entity Extraction & Validation (LLM + Natasha)
# ---------------------------------------
def extract_entities_from_chunk(text_chunk, openai_client):
    prompt = f"""
Extract the following entities from the text below:
- IP addresses
- Emails
- Company names
- Person names
- IBANs

Return them as valid JSON in the format:
{{
    "ip_addresses": [],
    "emails": [],
    "company_names": [],
    "person_names": [],
    "ibans": []
}}

If you see no valid data for a category, leave that list empty.

Text to analyze:
{text_chunk}
    """
    logger.debug("Extracting entities from chunk...")
    response_text = call_azure_openai_api(prompt, openai_client)
    logger.debug(f"Raw API response for entities: {response_text}")
    try:
        entities = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse JSON for entities: {e}. Response was: {response_text}")
        return {}
    
    ip_addresses = [ip for ip in entities.get("ip_addresses", []) if is_valid_ip(ip)]
    emails = [em for em in entities.get("emails", []) if is_valid_email(em)]
    ibans = [iban for iban in entities.get("ibans", []) if is_valid_iban(iban)]
    companies = entities.get("company_names", [])
    persons = entities.get("person_names", [])
    
    combined = {
        "ip_addresses": ip_addresses,
        "emails": emails,
        "company_names": companies,
        "person_names": persons,
        "ibans": ibans
    }
    combined = combine_entities(combined, text_chunk)
    logger.info(f"Extracted {len(ip_addresses)} IPs, {len(emails)} emails, and {len(persons)} person names from chunk.")
    return combined

# ---------------------------------------
# Fuzzy Deduplication for Entities
# ---------------------------------------
def unify_substring_entities(entity_set, threshold=90):
    entities = list(entity_set)
    entities.sort(key=len, reverse=True)
    final_list = []
    for candidate in entities:
        skip = False
        for i, existing in enumerate(final_list):
            similarity = fuzz.ratio(candidate.lower(), existing.lower())
            if similarity >= threshold:
                if len(candidate) > len(existing):
                    final_list[i] = candidate
                skip = True
                break
        if not skip:
            final_list.append(candidate)
    return set(final_list)

# ---------------------------------------
# Aggregation Pipeline for Entities
# ---------------------------------------
def aggregate_entities(chunks, openai_client):
    aggregated = {
        "ip_addresses": set(),
        "emails": set(),
        "company_names": set(),
        "person_names": set(),
        "ibans": set()
    }
    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk, openai_client)
        aggregated["ip_addresses"].update(entities.get("ip_addresses", []))
        aggregated["emails"].update(entities.get("emails", []))
        aggregated["company_names"].update(entities.get("company_names", []))
        aggregated["person_names"].update(entities.get("person_names", []))
        aggregated["ibans"].update(entities.get("ibans", []))
    aggregated["company_names"] = unify_substring_entities(aggregated["company_names"])
    aggregated["person_names"] = unify_substring_entities(aggregated["person_names"])
    return {k: sorted(list(v)) for k, v in aggregated.items()}

# ---------------------------------------
# Aggregation Pipeline with Frequency Counts
# ---------------------------------------
def aggregate_entities_with_counts(chunks, openai_client):
    freq = {
        "ip_addresses": Counter(),
        "emails": Counter(),
        "company_names": Counter(),
        "person_names": Counter(),
        "ibans": Counter()
    }
    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk, openai_client)
        for cat in freq.keys():
            for item in entities.get(cat, []):
                freq[cat][item] += 1
    return {cat: freq[cat].most_common() for cat in freq}

# ---------------------------------------
# Relationship Extraction
# ---------------------------------------
def extract_relationships_from_chunk(text_chunk, openai_client):
    prompt = f"""
Analyze the following text and extract relationships between the entities mentioned.

Allowed relationship labels: HAS_EMAIL, HAS_IP, IS_ASSOCIATED_WITH, IS_REGISTERED_AS, HAS_WEBSITE.
Any additional context should go under 'details'.
If a relationship is purely "Unknown - Unknown" or provides no meaningful information, omit it.

Return valid JSON in the format:
{{
  "relationships": [
    {{
      "source": "Entity Name",
      "relation": "Short Relationship Label (e.g. HAS_EMAIL, HAS_IP)",
      "target": "Related Entity or Value",
      "details": "Additional details if available (optional)"
    }}
  ]
}}

Text to analyze:
{text_chunk}
    """
    logger.debug("Extracting relationships from chunk...")
    response_text = call_azure_openai_api(prompt, openai_client)
    logger.debug(f"Raw API response for relationships: {response_text}")
    try:
        rels = json.loads(response_text)
        return rels.get("relationships", [])
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse JSON for relationships: {e}. Response was: {response_text}")
        return []

def aggregate_relationships(chunks, openai_client):
    relationships_list = []
    seen = set()
    for chunk in chunks:
        rels = extract_relationships_from_chunk(chunk, openai_client)
        for rel in rels:
            key = (
                rel.get("source", ""),
                rel.get("relation", ""),
                rel.get("target", ""),
                rel.get("details", "")
            )
            if key not in seen:
                seen.add(key)
                relationships_list.append(rel)
    return relationships_list

# ---------------------------------------
# CSV Saving Functions
# ---------------------------------------
def save_entities_to_csv(entities, csv_file_path):
    try:
        with open(csv_file_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["entity_type", "value"])
            for entity_type, values in entities.items():
                for val in values:
                    val_fixed = val.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                    writer.writerow([entity_type, val_fixed])
        logger.info(f"Entities saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving entities to CSV: {e}")

def save_relationships_to_csv(relationships, csv_file_path):
    try:
        with open(csv_file_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["source", "relation", "target", "details"])
            for rel in relationships:
                source = rel.get("source", "")
                relation = rel.get("relation", "")
                target = rel.get("target", "")
                details = rel.get("details", "")
                source_fixed = source.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                relation_fixed = relation.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                target_fixed = target.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                details_fixed = details.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                writer.writerow([source_fixed, relation_fixed, target_fixed, details_fixed])
        logger.info(f"Relationships saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving relationships to CSV: {e}")

# ---------------------------------------
# Visualization Functions
# ---------------------------------------
def visualize_relationships_pyvis(relationships):
    """
    Create an interactive PyVis graph of relationships and save as an HTML file.
    """
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.force_atlas_2based(gravity=-80, central_gravity=0.02, spring_length=100, spring_strength=0.01)
    for rel in relationships:
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        relation = rel.get("relation", "").strip()
        details = rel.get("details", "")
        pdf_id = rel.get("pdf_id", "")
        if source and target:
            net.add_node(source, label=source)
            net.add_node(target, label=target)
            title_str = f"{relation}: {details} (from {pdf_id})"
            net.add_edge(source, target, label=relation, title=title_str)
    net.show_buttons(filter_=['physics'])
    net.save_graph("pyvis_graph.html")
    return "pyvis_graph.html"

def visualize_top_entities(aggregated_counts, top_n=10):
    """
    Visualize top entities for each category as horizontal bar charts.
    """
    st.subheader("Top Entities by Category")
    for cat, items in aggregated_counts.items():
        if items:
            entities, counts = zip(*items)
            df = pd.DataFrame({"Entity": entities, "Count": counts})
            st.write(f"### {cat}")
            st.bar_chart(df.set_index("Entity"))
        else:
            st.write(f"### {cat} - No data")

# ---------------------------------------
# Aggregation Pipeline with Frequency Counts
# ---------------------------------------
def aggregate_entities_with_counts(chunks, openai_client):
    freq = {
        "ip_addresses": Counter(),
        "emails": Counter(),
        "company_names": Counter(),
        "person_names": Counter(),
        "ibans": Counter()
    }
    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk, openai_client)
        for cat in freq.keys():
            for item in entities.get(cat, []):
                freq[cat][item] += 1
    return {cat: freq[cat].most_common() for cat in freq}

# ---------------------------------------
# Main Pipeline for Extraction
# ---------------------------------------
def extract_data_from_pdf(pdf_file_path, openai_client, max_chars=1500):
    logger.info("Starting PDF extraction process...")
    raw_text = extract_text_from_pdf(pdf_file_path)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        logger.error("No text extracted from PDF.")
        return {}, []
    chunks = chunk_text(cleaned_text, max_chars=max_chars)
    entities = aggregate_entities(chunks, openai_client)
    relationships = aggregate_relationships(chunks, openai_client)
    # Tag each relationship with the PDF filename (for filtering)
    for rel in relationships:
        rel["pdf_id"] = os.path.basename(pdf_file_path)
    return entities, relationships, chunks

# ---------------------------------------
# Main App (Streamlit)
# ---------------------------------------
def app():
    st.title("PDF Relationship Viewer")

    # File uploader for PDF files
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.write("Please upload a PDF file.")
        return

    st.info("Processing PDF...")
    from openai import AzureOpenAI
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        default_headers={"Content-Type": "application/json"},
        api_version=API_VERSION,
    )
    entities, relationships, chunks = extract_data_from_pdf(pdf_file_path, openai_client, max_chars=1500)

    st.info("Extraction complete.")
    st.subheader("Extracted Entities:")
    st.json(entities)
    st.subheader("Extracted Relationships:")
    st.json(relationships)

    save_entities_to_csv(entities, "extracted_entities.csv")
    save_relationships_to_csv(relationships, "extracted_relationships.csv")

    # Build node type map from extracted entities for color-coding
    def build_node_type_map(entities):
        node_type_map = {}
        for ip in entities.get("ip_addresses", []):
            node_type_map[ip] = "ip"
        for em in entities.get("emails", []):
            node_type_map[em] = "email"
        for comp in entities.get("company_names", []):
            node_type_map[comp] = "company"
        for per in entities.get("person_names", []):
            node_type_map[per] = "person"
        for iban in entities.get("ibans", []):
            node_type_map[iban] = "iban"
        return node_type_map

    node_type_map = build_node_type_map(entities)

    # Sidebar filtering options for relationships
    pdf_ids = sorted({rel["pdf_id"] for rel in relationships})
    relation_types = sorted({rel["relation"] for rel in relationships})
    st.sidebar.header("Relationship Filters")
    selected_pdfs = st.sidebar.multiselect("Select PDFs to include:", options=pdf_ids, default=pdf_ids)
    selected_types = st.sidebar.multiselect("Select relationship types:", options=relation_types, default=relation_types)
    search_text = st.sidebar.text_input("Search for entity (source or target):", "")
    filtered_rels = []
    for rel in relationships:
        if selected_pdfs and rel.get("pdf_id", "") not in selected_pdfs:
            continue
        if selected_types and rel.get("relation", "") not in selected_types:
            continue
        if search_text:
            stxt = search_text.lower()
            if stxt not in rel.get("source", "").lower() and stxt not in rel.get("target", "").lower():
                continue
        filtered_rels.append(rel)
    st.write(f"Showing {len(filtered_rels)} relationships based on filters.")

    # Visualize top entities counts
    aggregated_counts = aggregate_entities_with_counts(chunks, openai_client)
    visualize_top_entities(aggregated_counts, top_n=10)

    # Build and display the interactive PyVis graph if there are relationships
    if filtered_rels:
        html_file = visualize_relationships_pyvis(filtered_rels)
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800)
    else:
        st.write("No relationships match your filters.")

if __name__ == "__main__":
    app()
