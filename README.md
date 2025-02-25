EUCOM LLM

EUCOM LLM is a Streamlit application designed to extract, validate, and visualize entities and relationships from PDF documents. The application leverages Azure OpenAI for natural language processing, integrates Russian NLP via Natasha, and includes several validation utilities for common data types like IP addresses, emails, phone numbers, and IBANs.

Features

PDF Extraction:
Extracts text from PDF files using pdfminer.
Entity Extraction:
Uses Azure OpenAI to extract entities such as IP addresses, emails, company names, person names, and IBANs from text.
Russian NLP Integration:
Enhances entity recognition by using Natasha to detect Russian names and organizations.
Relationship Extraction:
Extracts relationships between entities (e.g., HAS_EMAIL, HAS_IP) and provides additional contextual details.
Fuzzy Deduplication:
Applies fuzzy matching to deduplicate similar entities.
Visualization:
Visualizes relationships using interactive graphs (via PyVis) and bar charts for entity frequency (via Streamlit and Matplotlib).
CSV Export:
Saves extracted entities and relationships as CSV files for further analysis.
