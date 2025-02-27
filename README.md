# MERET - MITRE Entity Relationship Extraction Tool

MERET is a Streamlit application designed to extract, validate, and visualize entities and relationships from PDF documents. The application leverages Azure OpenAI for natural language processing, integrates Russian NLP via Natasha, and includes several validation utilities for common data types like IP addresses, emails, phone numbers, and IBANs.

## Features

### Document Processing
- **Multiple PDF Support**: Upload and process multiple PDF documents simultaneously
- **PDF Extraction**: Extracts text from PDF files using pdfminer
- **Incremental Processing**: Maintains state between sessions for continuous analysis

### Entity Extraction
- **Advanced Entity Recognition**: Uses Azure OpenAI to extract entities such as IP addresses, emails, company names, person names, IBANs, and geographic locations
- **Russian NLP Integration**: Enhances entity recognition by using Natasha to detect Russian names and organizations
- **Fuzzy Deduplication**: Applies fuzzy matching to deduplicate similar entities
- **Entity Frequency Analysis**: Tracks and displays entity frequency across documents

### Relationship Analysis
- **Relationship Extraction**: Extracts relationships between entities (e.g., HAS_EMAIL, HAS_IP) and provides additional contextual details
- **Cross-Document Relationships**: Tracks relationships across multiple documents
- **Relationship Filtering**: Filter relationships by document, relationship type, or entity name
- **Sentiment Analysis**: Analyzes sentiment for entity mentions across documents

### Visualization
- **Interactive Network Graphs**: Visualizes relationships using interactive graphs (via PyVis)
- **Entity Frequency Charts**: Displays bar charts for entity frequency (via Streamlit and Matplotlib)
- **Sentiment Visualization**: Displays sentiment analysis for entities with positive/negative indicators
- **Geographic Mapping**: Visualizes location entities on interactive maps with sentiment analysis
- **Customizable Filtering**: Filter visualizations based on document source, relationship type, or search terms

### Export Options
- **CSV Export**: Exports extracted entities and relationships as CSV files
- **Analyst Notebook Support**: Exports data in i2 Analyst's Notebook XML format for further analysis in professional intelligence tools
- **Interactive Downloads**: Direct download buttons for all export formats

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app3.py
```

2. Upload one or more PDF documents through the sidebar
3. View extracted entities and relationships
4. Filter and visualize the data as needed
5. Export the results in your preferred format
