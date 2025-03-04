import json
import csv
import logging
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from textblob import TextBlob

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

# Initialize geocoder with a timeout
geolocator = Nominatim(user_agent="meret-entity-extractor", timeout=10)

# Cache for geocoding results to avoid repeated API calls
geo_cache = {}

# Initialize NLTK data for TextBlob
try:
    import nltk
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

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
- Locations (cities, countries, regions)

Return them as valid JSON in the format:
{{
    "ip_addresses": [],
    "emails": [],
    "company_names": [],
    "person_names": [],
    "ibans": [],
    "locations": []
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
    locations = entities.get("locations", [])
    
    combined = {
        "ip_addresses": ip_addresses,
        "emails": emails,
        "company_names": companies,
        "person_names": persons,
        "ibans": ibans,
        "locations": locations
    }
    combined = combine_entities(combined, text_chunk)
    logger.info(f"Extracted {len(ip_addresses)} IPs, {len(emails)} emails, {len(persons)} person names, and {len(locations)} locations from chunk.")
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
# Sentiment Analysis Functions
# ---------------------------------------
def analyze_entity_sentiment(entity, text_chunks):
    """
    Analyze sentiment for an entity across all text chunks
    Returns a sentiment score (-1 to 1) and count of mentions
    Uses a simpler approach to avoid TextBlob sentence tokenization issues
    """
    sentiment_sum = 0.0
    mention_count = 0
    
    for chunk in text_chunks:
        # Count occurrences in this chunk
        lower_chunk = chunk.lower()
        lower_entity = entity.lower()
        
        if lower_entity in lower_chunk:
            # Simple approach: analyze sentiment for the whole chunk
            # if it contains the entity
            try:
                blob = TextBlob(chunk)
                sentiment_sum += blob.sentiment.polarity
                mention_count += 1
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {e}")
    
    # Calculate average sentiment if there are mentions
    if mention_count > 0:
        return {
            "sentiment": sentiment_sum / mention_count,
            "mentions": mention_count
        }
    else:
        return {
            "sentiment": 0,
            "mentions": 0
        }

def get_entity_locations(locations):
    """
    Get geolocation data for location entities
    Returns a dictionary of locations with coordinates
    Handles geocoding errors gracefully
    """
    geo_results = {}
    
    # Simplify by providing a few common locations manually to avoid API calls
    common_locations = {
        "moscow": {"latitude": 55.7558, "longitude": 37.6173, "address": "Moscow, Russia"},
        "new york": {"latitude": 40.7128, "longitude": -74.0060, "address": "New York, NY, USA"},
        "london": {"latitude": 51.5074, "longitude": -0.1278, "address": "London, UK"},
        "berlin": {"latitude": 52.5200, "longitude": 13.4050, "address": "Berlin, Germany"},
        "paris": {"latitude": 48.8566, "longitude": 2.3522, "address": "Paris, France"},
        "tokyo": {"latitude": 35.6762, "longitude": 139.6503, "address": "Tokyo, Japan"},
        "beijing": {"latitude": 39.9042, "longitude": 116.4074, "address": "Beijing, China"},
        "washington": {"latitude": 38.9072, "longitude": -77.0369, "address": "Washington, DC, USA"}
    }
    
    # Update our cache with common locations
    for loc, data in common_locations.items():
        if loc not in geo_cache:
            geo_cache[loc] = data
    
    for location in locations:
        location_lower = location.lower()
        
        # Skip if already in cache
        if location in geo_cache:
            geo_results[location] = geo_cache[location]
            continue
        
        # Check if a common location is contained in this location name
        found = False
        for common_loc in common_locations:
            if common_loc in location_lower:
                geo_results[location] = common_locations[common_loc]
                geo_cache[location] = common_locations[common_loc]
                found = True
                break
                
        if found:
            continue
        
        # If not found in common locations, try geocoding
        try:
            # Try to geocode the location
            geo_location = geolocator.geocode(location, timeout=5)
            if geo_location:
                result = {
                    "latitude": geo_location.latitude,
                    "longitude": geo_location.longitude,
                    "address": geo_location.address
                }
                # Cache the result
                geo_cache[location] = result
                geo_results[location] = result
                logger.info(f"Geocoded location: {location}")
            else:
                logger.warning(f"Could not geocode location: {location}")
        except Exception as e:
            logger.error(f"Error geocoding location {location}: {e}")
    
    return geo_results

# ---------------------------------------
# Aggregation Pipeline for Entities
# ---------------------------------------
def aggregate_entities(chunks, openai_client):
    aggregated = {
        "ip_addresses": set(),
        "emails": set(),
        "company_names": set(),
        "person_names": set(),
        "ibans": set(),
        "locations": set()
    }
    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk, openai_client)
        aggregated["ip_addresses"].update(entities.get("ip_addresses", []))
        aggregated["emails"].update(entities.get("emails", []))
        aggregated["company_names"].update(entities.get("company_names", []))
        aggregated["person_names"].update(entities.get("person_names", []))
        aggregated["ibans"].update(entities.get("ibans", []))
        aggregated["locations"].update(entities.get("locations", []))
    
    # Apply deduplication
    aggregated["company_names"] = unify_substring_entities(aggregated["company_names"])
    aggregated["person_names"] = unify_substring_entities(aggregated["person_names"])
    aggregated["locations"] = unify_substring_entities(aggregated["locations"])
    
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
        "ibans": Counter(),
        "locations": Counter()
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
# CSV Functions
# ---------------------------------------
def entities_to_csv(entities):
    """
    Convert entities to CSV string for download
    """
    try:
        # Use StringIO for text data in memory
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["entity_type", "value"])
        
        # Write each entity to the CSV
        for entity_type, values in entities.items():
            for val in values:
                # Handle potential encoding issues
                val_fixed = str(val).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                writer.writerow([entity_type, val_fixed])
        
        # Get the CSV content
        csv_content = output.getvalue()
        logger.info(f"Generated CSV with {sum(len(values) for values in entities.values())} entity rows")
        return csv_content
    except Exception as e:
        logger.error(f"Error converting entities to CSV: {e}")
        st.error(f"Error generating CSV: {e}")
        return "entity_type,value\nerror,error generating CSV"

def relationships_to_csv(relationships):
    """
    Convert relationships to CSV string for download
    """
    try:
        # Use StringIO for text data in memory
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["source", "relation", "target", "details", "pdf_id"])
        
        row_count = 0
        # Write each relationship to the CSV
        for rel in relationships:
            source = rel.get("source", "")
            relation = rel.get("relation", "")
            target = rel.get("target", "")
            details = rel.get("details", "")
            pdf_id = rel.get("pdf_id", "")
            
            # Handle potential encoding issues
            source_fixed = str(source).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            relation_fixed = str(relation).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            target_fixed = str(target).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            details_fixed = str(details).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            pdf_id_fixed = str(pdf_id).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            writer.writerow([source_fixed, relation_fixed, target_fixed, details_fixed, pdf_id_fixed])
            row_count += 1
        
        # Get the CSV content
        csv_content = output.getvalue()
        logger.info(f"Generated CSV with {row_count} relationship rows")
        return csv_content
    except Exception as e:
        logger.error(f"Error converting relationships to CSV: {e}")
        st.error(f"Error generating CSV: {e}")
        return "source,relation,target,details,pdf_id\nerror,error,error,error generating CSV,error"

# For backward compatibility
def save_entities_to_csv(entities, csv_file_path):
    """
    Save entities to a CSV file on disk
    """
    try:
        with open(csv_file_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["entity_type", "value"])
            for entity_type, values in entities.items():
                for val in values:
                    val_fixed = str(val).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                    writer.writerow([entity_type, val_fixed])
        logger.info(f"Entities saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving entities to CSV: {e}")

# For backward compatibility
def save_relationships_to_csv(relationships, csv_file_path):
    """
    Save relationships to a CSV file on disk
    """
    try:
        with open(csv_file_path, mode="w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["source", "relation", "target", "details", "pdf_id"])
            for rel in relationships:
                source = rel.get("source", "")
                relation = rel.get("relation", "")
                target = rel.get("target", "")
                details = rel.get("details", "")
                pdf_id = rel.get("pdf_id", "")
                
                source_fixed = str(source).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                relation_fixed = str(relation).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                target_fixed = str(target).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                details_fixed = str(details).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                
                writer.writerow([source_fixed, relation_fixed, target_fixed, details_fixed, pdf_id])
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
# Main Pipeline for Extraction
# ---------------------------------------
def extract_data_from_pdf(pdf_file_path, openai_client, max_chars=1500):
    logger.info(f"Starting PDF extraction process for {pdf_file_path}...")
    raw_text = extract_text_from_pdf(pdf_file_path)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        logger.error("No text extracted from PDF.")
        return {}, [], []
    chunks = chunk_text(cleaned_text, max_chars=max_chars)
    entities = aggregate_entities(chunks, openai_client)
    relationships = aggregate_relationships(chunks, openai_client)
    # Tag each relationship with the PDF filename (for filtering)
    for rel in relationships:
        rel["pdf_id"] = os.path.basename(pdf_file_path)
    return entities, relationships, chunks


# ---------------------------------------
# Export to Analyst Notebook Functions
# ---------------------------------------
def export_to_analyst_notebook(entities, relationships, file_path="analyst_notebook_export.xml"):
    """
    Export entities and relationships to Analyst Notebook compatible XML format.
    Based on i2 Analyst's Notebook XML import format.
    """
    try:
        # Create root element
        root = ET.Element("ANALYSIS_DATA")
        root.set("Version", "4")
        
        # Create entities section
        entities_element = ET.SubElement(root, "ENTITIES")
        
        # Counter for unique IDs
        entity_counter = 1
        entity_id_map = {}  # Map entity names to their IDs
        
        # Process each entity type
        for entity_type, values in entities.items():
            icon_type = {
                "person_names": "ET_PERSON",
                "company_names": "ET_ORGANIZATION",
                "ip_addresses": "ET_NETWORK_IPV4_ADDRESS",
                "emails": "ET_EMAIL_ADDRESS",
                "ibans": "ET_BANK_ACCOUNT",
                "locations": "ET_LOCATION"
            }.get(entity_type, "ET_THING")
            
            for value in values:
                entity = ET.SubElement(entities_element, "ENTITY")
                entity.set("EntityId", f"e{entity_counter}")
                entity_id_map[value] = f"e{entity_counter}"
                entity_counter += 1
                
                # Set entity attributes
                ET.SubElement(entity, "IDENTITY").text = value
                ET.SubElement(entity, "TYPE").text = icon_type
                ET.SubElement(entity, "LABEL").text = value
                # Add additional attributes if needed
                props = ET.SubElement(entity, "PROPERTIES")
                ET.SubElement(props, "PROPERTY", Name="Entity Type").text = entity_type.replace("_", " ").title()
        
        # Create links section
        links_element = ET.SubElement(root, "LINKS")
        
        # Process relationships
        link_counter = 1
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            
            # Skip if source or target not in our entity map
            if source not in entity_id_map or target not in entity_id_map:
                continue
                
            link = ET.SubElement(links_element, "LINK")
            link.set("LinkId", f"l{link_counter}")
            link_counter += 1
            
            # Set link attributes
            ET.SubElement(link, "IDENTITY").text = rel.get("relation", "")
            ET.SubElement(link, "TYPE").text = "LT_ASSOCIATION"
            ET.SubElement(link, "LABEL").text = rel.get("relation", "")
            ET.SubElement(link, "END1", EntityId=entity_id_map[source], EntityLocationX="0", EntityLocationY="0")
            ET.SubElement(link, "END2", EntityId=entity_id_map[target], EntityLocationX="0", EntityLocationY="0")
            
            # Add details as property if available
            if rel.get("details"):
                props = ET.SubElement(link, "PROPERTIES")
                ET.SubElement(props, "PROPERTY", Name="Details").text = rel.get("details", "")
                ET.SubElement(props, "PROPERTY", Name="Source").text = rel.get("pdf_id", "")
        
        # Create XML string instead of saving to file
        from io import StringIO
        xml_output = StringIO()
        tree = ET.ElementTree(root)
        
        # First save to a temporary file
        temp_path = file_path
        tree.write(temp_path, encoding="utf-8", xml_declaration=True)
        
        # Read the file back as a string
        with open(temp_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        logger.info(f"Generated Analyst Notebook XML with {entity_counter-1} entities and {link_counter-1} links")
        return temp_path
    except Exception as e:
        logger.error(f"Error exporting to Analyst Notebook format: {e}")
        st.error(f"Error generating Analyst Notebook XML: {e}")
        return None

# ---------------------------------------
# Main App (Streamlit)
# ---------------------------------------
def app():
    # For visualizing sentiment analysis
    def visualize_entity_sentiment(entity_type, entities, chunks):
        st.subheader(f"Sentiment Analysis for {entity_type.replace('_', ' ').title()}")
        if not entities:
            st.info(f"No {entity_type} found for sentiment analysis")
            return
            
        sentiment_data = []
        for entity in entities:
            sentiment_info = analyze_entity_sentiment(entity, chunks)
            if sentiment_info["mentions"] > 0:
                sentiment_data.append({
                    "Entity": entity,
                    "Sentiment": sentiment_info["sentiment"],
                    "Mentions": sentiment_info["mentions"]
                })
        
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Sort by sentiment for better visualization
            sentiment_df = sentiment_df.sort_values("Sentiment")
            
            # Create color map based on sentiment
            colors = ['#d32f2f' if s < -0.1 else '#2e7d32' if s > 0.1 else '#ffa000' 
                    for s in sentiment_df["Sentiment"]]
                    
            # Plot sentiment chart
            fig, ax = plt.subplots(figsize=(10, max(5, len(sentiment_df) * 0.3)))
            bars = ax.barh(sentiment_df["Entity"], sentiment_df["Sentiment"], color=colors)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Negative <--> Positive")
            ax.set_title(f"Sentiment Analysis for {entity_type.replace('_', ' ').title()}")
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display as table too
            sentiment_df["Sentiment"] = sentiment_df["Sentiment"].map(lambda x: f"{x:.2f}")
            st.dataframe(sentiment_df)
        else:
            st.info("No sentiment data available")
    
    # For visualizing geo locations
    def visualize_locations(locations, chunks):
        st.subheader("Geographic Entities")
        if not locations:
            st.info("No locations found for mapping")
            return
        
        try:
            # Get geo data
            geo_data = get_entity_locations(locations)
            
            if not geo_data:
                st.warning("Could not geocode any locations")
                return
                
            # Create sentiment info for locations
            loc_with_sentiment = []
            for loc in geo_data:
                sentiment_info = analyze_entity_sentiment(loc, chunks)
                sentiment = sentiment_info["sentiment"]
                mentions = sentiment_info["mentions"]
                loc_with_sentiment.append({
                    "location": loc,
                    "lat": geo_data[loc]["latitude"],
                    "lon": geo_data[loc]["longitude"],
                    "sentiment": sentiment,
                    "mentions": mentions,
                    "address": geo_data[loc]["address"]
                })
            
            if loc_with_sentiment:
                # Create a safer display of the data as a table first
                st.subheader("Location Data")
                loc_df = pd.DataFrame(loc_with_sentiment)
                loc_df["sentiment"] = loc_df["sentiment"].map(lambda x: f"{x:.2f}")
                st.dataframe(loc_df[["location", "sentiment", "mentions", "address"]])
                
                try:
                    # Create map centered on average coordinates
                    avg_lat = sum(item["lat"] for item in loc_with_sentiment) / len(loc_with_sentiment)
                    avg_lon = sum(item["lon"] for item in loc_with_sentiment) / len(loc_with_sentiment)
                    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)
                    
                    # Add markers for each location
                    for item in loc_with_sentiment:
                        # Determine color based on sentiment
                        if item["sentiment"] < -0.1:
                            color = "red"
                        elif item["sentiment"] > 0.1:
                            color = "green"
                        else:
                            color = "orange"
                        
                        # Create popup content
                        popup_content = f"""
                        <b>{item['location']}</b><br>
                        Sentiment: {float(item['sentiment']):.2f}<br>
                        Mentions: {item['mentions']}<br>
                        Address: {item['address']}
                        """
                        
                        # Add marker
                        folium.Marker(
                            location=[item["lat"], item["lon"]],
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=item["location"],
                            icon=folium.Icon(color=color)
                        ).add_to(m)
                    
                    # Display map
                    st.subheader("Location Map")
                    folium_static(m)
                except Exception as e:
                    st.error(f"Error displaying map: {e}")
                    logger.error(f"Map display error: {e}")
        except Exception as e:
            st.error(f"Error processing location data: {e}")
            logger.error(f"Location visualization error: {e}")
            
    st.title("MERET - MITRE Entity Relationship Extraction Tool")

    # Initialize session state for storing multiple PDF data
    if 'all_entities' not in st.session_state:
        st.session_state.all_entities = {}
    if 'all_relationships' not in st.session_state:
        st.session_state.all_relationships = []
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = {}
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = []
    if 'combined_entities' not in st.session_state:
        st.session_state.combined_entities = {
            "ip_addresses": [],
            "emails": [],
            "company_names": [],
            "person_names": [],
            "ibans": [],
            "locations": []
        }

    # File uploader for multiple PDF files
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Initialize OpenAI client
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        default_headers={"Content-Type": "application/json"},
        api_version=API_VERSION,
    )
    
    # Process uploaded PDFs if any
    if uploaded_files:
        os.makedirs("temp", exist_ok=True)
        
        # Identify new PDFs to process
        new_pdfs = []
        for uploaded_file in uploaded_files:
            pdf_name = uploaded_file.name
            if pdf_name not in st.session_state.processed_pdfs:
                new_pdfs.append(uploaded_file)
                
        # Process new PDFs if any
        if new_pdfs:
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(new_pdfs):
                pdf_name = uploaded_file.name
                st.info(f"Processing {pdf_name}...")
                
                # Save file temporarily
                pdf_file_path = os.path.join("temp", pdf_name)
                with open(pdf_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract data
                entities, relationships, chunks = extract_data_from_pdf(pdf_file_path, openai_client, max_chars=1500)
                
                # Store results
                st.session_state.all_entities[pdf_name] = entities
                st.session_state.all_relationships.extend(relationships)
                st.session_state.all_chunks[pdf_name] = chunks
                st.session_state.processed_pdfs.append(pdf_name)
                
                # Update progress
                progress_bar.progress((i + 1) / len(new_pdfs))
            
            # Combine all entities
            combined = {
                "ip_addresses": [],
                "emails": [],
                "company_names": [],
                "person_names": [],
                "ibans": [],
                "locations": []
            }
            
            for pdf_entities in st.session_state.all_entities.values():
                for key in combined:
                    if key in pdf_entities:
                        combined[key].extend(pdf_entities.get(key, []))
            
            # Deduplicate combined entities
            for key in combined:
                combined[key] = list(set(combined[key]))
                
            st.session_state.combined_entities = combined
            st.success(f"Processed {len(new_pdfs)} new PDFs.")
    
    # If no PDFs are processed yet, show a message and return
    if not st.session_state.processed_pdfs:
        st.write("Please upload PDF files to analyze.")
        return
    
    # Display summary of processed PDFs
    st.subheader("Processed PDFs:")
    st.write(", ".join(st.session_state.processed_pdfs))
    
    # Show combined entities
    with st.expander("View Combined Extracted Entities", expanded=False):
        st.json(st.session_state.combined_entities)
    
    # Export buttons section
    st.subheader("Export Options")
    
    # Check if there's data to export
    has_entities = any(len(entities) > 0 for entities in st.session_state.combined_entities.values())
    has_relationships = len(st.session_state.all_relationships) > 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate CSV data for entities
        if has_entities:
            entities_csv = entities_to_csv(st.session_state.combined_entities)
            st.download_button(
                label=f"Export Entities to CSV ({sum(len(v) for v in st.session_state.combined_entities.values())} items)",
                data=entities_csv,
                file_name="entity_export.csv",
                mime="text/csv",
                help="Download entities as a CSV file"
            )
        else:
            st.info("No entities to export yet")
    
    with col2:
        if has_entities and has_relationships:
            xml_path = export_to_analyst_notebook(
                st.session_state.combined_entities, 
                st.session_state.all_relationships
            )
            if xml_path:
                with open(xml_path, "r", encoding="utf-8") as xml_file:
                    xml_content = xml_file.read()
                    st.download_button(
                        label="Export to Analyst Notebook",
                        data=xml_content,
                        file_name="analyst_notebook_export.xml",
                        mime="application/xml",
                        help="Download data in i2 Analyst's Notebook format"
                    )
        else:
            st.info("Need both entities and relationships for Analyst Notebook export")
    
    # Relationships CSV export
    if has_relationships:
        relationships_csv = relationships_to_csv(st.session_state.all_relationships)
        st.download_button(
            label=f"Export Relationships to CSV ({len(st.session_state.all_relationships)} items)",
            data=relationships_csv,
            file_name="relationships_export.csv",
            mime="text/csv",
            help="Download relationships as a CSV file"
        )
    else:
        st.info("No relationships to export yet")
    
    # Build node type map from combined entities for color-coding
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
        for loc in entities.get("locations", []):
            node_type_map[loc] = "location"
        return node_type_map

    node_type_map = build_node_type_map(st.session_state.combined_entities)

    # Sidebar filtering options for relationships
    pdf_ids = sorted({rel["pdf_id"] for rel in st.session_state.all_relationships})
    relation_types = sorted({rel["relation"] for rel in st.session_state.all_relationships})
    st.sidebar.header("Relationship Filters")
    selected_pdfs = st.sidebar.multiselect("Select PDFs to include:", options=pdf_ids, default=pdf_ids)
    selected_types = st.sidebar.multiselect("Select relationship types:", options=relation_types, default=relation_types)
    search_text = st.sidebar.text_input("Search for entity (source or target):", "")
    
    # Filter relationships based on selections
    filtered_rels = []
    for rel in st.session_state.all_relationships:
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

    # Collect all chunks for aggregation and sentiment analysis
    all_chunks_list = []
    for chunks_list in st.session_state.all_chunks.values():
        all_chunks_list.extend(chunks_list)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Network Graph", "Entity Frequency", "Sentiment Analysis", "Geo Mapping"])
    
    with tab1:
        # Build and display the interactive PyVis graph if there are relationships
        if filtered_rels:
            html_file = visualize_relationships_pyvis(filtered_rels)
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800)
        else:
            st.write("No relationships match your filters.")
    
    with tab2:
        # Visualize top entities counts
        aggregated_counts = aggregate_entities_with_counts(all_chunks_list, openai_client)
        visualize_top_entities(aggregated_counts, top_n=10)
    
    with tab3:
        # Allow user to select entity types for sentiment analysis
        entity_type_options = [
            ("person_names", "Person Names"),
            ("company_names", "Company Names"),
            ("locations", "Locations")
        ]
        
        selected_entity_type = st.selectbox(
            "Select entity type for sentiment analysis:",
            options=[et[0] for et in entity_type_options],
            format_func=lambda x: dict(entity_type_options)[x]
        )
        
        # Run sentiment analysis on selected entity type
        entities_for_sentiment = st.session_state.combined_entities.get(selected_entity_type, [])
        if entities_for_sentiment:
            visualize_entity_sentiment(selected_entity_type, entities_for_sentiment, all_chunks_list)
        else:
            st.info(f"No {dict(entity_type_options)[selected_entity_type]} found for sentiment analysis")
    
    with tab4:
        # Visualize locations on a map
        locations = st.session_state.combined_entities.get("locations", [])
        visualize_locations(locations, all_chunks_list)

if __name__ == "__main__":
    app()