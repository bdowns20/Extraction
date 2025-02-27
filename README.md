# EUCOM LLM

EUCOM LLM is a Streamlit application designed to extract, validate, and visualize entities and relationships from PDF documents. The application leverages Azure OpenAI for natural language processing, integrates Russian NLP via Natasha, and includes several validation utilities for common data types like IP addresses, emails, phone numbers, and IBANs.

## Features

### Document Processing
- **Multiple PDF Support**: Upload and process multiple PDF documents simultaneously
- **PDF Extraction**: Extracts text from PDF files using pdfminer
- **Incremental Processing**: Maintains state between sessions for continuous analysis

### Entity Extraction
- **Advanced Entity Recognition**: Uses Azure OpenAI to extract entities such as IP addresses, emails, company names, person names, and IBANs
- **Russian NLP Integration**: Enhances entity recognition by using Natasha to detect Russian names and organizations
- **Fuzzy Deduplication**: Applies fuzzy matching to deduplicate similar entities
- **Entity Frequency Analysis**: Tracks and displays entity frequency across documents

### Relationship Analysis
- **Relationship Extraction**: Extracts relationships between entities (e.g., HAS_EMAIL, HAS_IP) and provides additional contextual details
- **Cross-Document Relationships**: Tracks relationships across multiple documents
- **Relationship Filtering**: Filter relationships by document, relationship type, or entity name

### Visualization
- **Interactive Network Graphs**: Visualizes relationships using interactive graphs (via PyVis)
- **Entity Frequency Charts**: Displays bar charts for entity frequency (via Streamlit and Matplotlib)
- **Customizable Filtering**: Filter visualizations based on document source, relationship type, or search terms

### Export Options
- **CSV Export**: Saves extracted entities and relationships as CSV files
- **PDF Reports**: Generates formatted PDF reports of extracted entities
- **Analyst Notebook Support**: Exports data in i2 Analyst's Notebook XML format for further analysis in professional intelligence tools
- **Interactive Downloads**: Direct download buttons for all export formats

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app2.py
```

2. Upload one or more PDF documents through the sidebar
3. View extracted entities and relationships
4. Filter and visualize the data as needed
5. Export the results in your preferred format
