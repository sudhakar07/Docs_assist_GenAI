import streamlit as st
import google.generativeai as genai
import traceback
from typing import List, Dict, Any
import re
# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Presidio Imports
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

import os
# pip install llama-index-llms-gemini llama-index-embeddings-gemini llama-index-postprocessor-presidio

# pip install presidio-analyzer presidio-anonymizer
genai.configure(api_key='AIzaSyChNUtW6XZ5lUiWnFqU4SgcOEwLKeLq8q4')
os.environ["GOOGLE_API_KEY"] = 'AIzaSyChNUtW6XZ5lUiWnFqU4SgcOEwLKeLq8q4'

class PIISafeRAGApplication:
    def __init__(self):
        # Configure Gemini API
        # try:
        #     genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))
        # except Exception as e:
        #     st.error(f"Error configuring Gemini API: {e}")
        
        # Comprehensive list of PII entities to mask
        self.pii_entities = [
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", 
            "CREDIT_CARD", "CRYPTO", "IP_ADDRESS",
            "LOCATION", "COUNTRY", "CITY", "STATE", 
            "STREET_ADDRESS", "POSTAL_CODE",
            "NRP", "SSN", "MEDICAL_LICENSE", 
            "US_PASSPORT", "DRIVER_LICENSE", 
            "BANK_ACCOUNT", "IBAN", 
            "ORGANIZATION", "JOB_TITLE", 
            "DOMAIN_NAME", "URL", 
            "MAC_ADDRESS", "BITCOIN_ADDRESS", 
            "DATE_OF_BIRTH", "AGE", "GENDER"
        ]
        
        # Initialize Presidio Analyzer
        self.analyzer = AnalyzerEngine()
        
        # Custom entities dictionary to store user-defined entities
        if 'custom_entities' not in st.session_state:
            st.session_state.custom_entities = {}

        # Initialize LlamaIndex Settings
        try:
            Settings.llm = Gemini(model_name="models/gemini-pro")
            Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
            Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
        except Exception as e:
            st.error(f"Error initializing LlamaIndex settings: {e}")
        
        # Session state management
        if 'original_text' not in st.session_state:
            st.session_state.original_text = ""
        if 'masked_text' not in st.session_state:
            st.session_state.masked_text = ""
        if 'index' not in st.session_state:
            st.session_state.index = None
        if 'detected_entities' not in st.session_state:
            st.session_state.detected_entities = []

    def add_custom_entity(self, entity_name: str, regex_pattern: str):
        """Add a custom entity with its regex pattern"""
        try:
            # Validate regex
            re.compile(regex_pattern)
            
            # Add to session state custom entities
            st.session_state.custom_entities[entity_name] = regex_pattern
            
            # Create a custom pattern recognizer
            pattern = Pattern(
                name=entity_name,
                regex=regex_pattern,
                score=0.5
            )
            custom_recognizer = PatternRecognizer(
                supported_entity=entity_name,
                patterns=[pattern]
            )
            
            # Add the custom recognizer to the analyzer
            self.analyzer.registry.add_recognizer(custom_recognizer)
            
            st.success(f"Custom entity '{entity_name}' added successfully!")
        except re.error:
            st.error("Invalid regex pattern. Please check your input.")
        except Exception as e:
            st.error(f"Error adding custom entity: {e}")

    def detect_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in the text with versioned masking"""
        try:
            # Combine standard and custom entity types
            st.write(list(st.session_state.custom_entities.keys()))
            all_entities = self.pii_entities + list(st.session_state.custom_entities.keys())
            
            # Analyze text for PII entities
            results = self.analyzer.analyze(
                text=text, 
                entities=all_entities,
                language='en'
            )
            
            # Create a dictionary to track entity counts
            entity_counts = {}
            
            # Create a list of detected entities
            detected_entities = []
            for result in results:
                # Get the base entity type
                base_type = result.entity_type
                
                # Increment and track the count for this entity type
                entity_counts[base_type] = entity_counts.get(base_type, 0) + 1
                
                # Create a versioned entity type
                versioned_type = f"{base_type}_{entity_counts[base_type]}"
                
                # Check if this exact entity has already been added to avoid duplicates
                existing_entities = [
                    e for e in detected_entities 
                    if e['original_value'] == text[result.start:result.end]
                ]
                
                if not existing_entities:
                    detected_entities.append({
                        "entity_type": versioned_type,
                        "base_type": base_type,
                        "original_value": text[result.start:result.end],
                        "masked_value": f"[{versioned_type}]",
                        "start": result.start,
                        "end": result.end
                    })
            
            return detected_entities
        except Exception as e:
            st.error(f"Error detecting PII entities: {e}")
            return []

    def process_text(self, text: str):
        """Process text with comprehensive PII masking"""
        try:
            # Save original text
            st.session_state.original_text = text
            
            # Detect PII entities
            detected_entities = self.detect_pii_entities(text)
            st.session_state.detected_entities = detected_entities
            
            # Mask the text
            masked_text = self._mask_text(text, detected_entities)
            st.session_state.masked_text = masked_text
            
            # Create LlamaIndex document with masked text
            document = Document(text=masked_text)
            
            # Create vector index
            index = VectorStoreIndex.from_documents(
                [document], 
                transformations=[Settings.node_parser]
            )
            
            # Store index in session state
            st.session_state.index = index
            
            return masked_text
        
        except Exception as e:
            st.error(f"Error processing text: {e}")
            st.error(traceback.format_exc())
            return text

    def _mask_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Manually mask text based on detected entities"""
        # Sort entities by start position in reverse to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            # Replace the detected entity with its masked value
            masked_text = (
                masked_text[:entity['start']] + 
                entity['masked_value'] + 
                masked_text[entity['end']:]
            )
        
        return masked_text

    def _unmask_response(self, masked_response: str) -> str:
        """Replace masked entities with original values in the response"""
        unmasked_response = masked_response
        
        # Sort entities to ensure correct replacement
        sorted_entities = sorted(
            st.session_state.detected_entities, 
            key=lambda x: len(x['masked_value']), 
            reverse=True
        )
        
        # Replace masked entities with original values
        for entity in sorted_entities:
            unmasked_response = unmasked_response.replace(
                entity['masked_value'], 
                entity['original_value']
            )
        
        return unmasked_response

    def query_rag(self, query: str):
        """Perform RAG query with PII protection"""
        try:
            if not st.session_state.index:
                st.error("Please load a document first!")
                return ""
            
            # Create query engine 
            query_engine = st.session_state.index.as_query_engine()
            
            # Execute query on masked text
            masked_response = query_engine.query(query)
            st.sidebar.write(f"masked_response",masked_response)
            
            
            # Unmask the response
            unmasked_response = self._unmask_response(str(masked_response))
            
            return unmasked_response
        
        except Exception as e:
            st.error(f"Error performing RAG query: {e}")
            return ""

    def render_custom_entity_tab(self):
        """Render the Custom Entity Configuration Tab"""
        st.header("Custom Entity Configuration")
        
        # Input for new custom entity
        col1, col2 = st.columns(2)
        with col1:
            new_entity_name = st.text_input(
                "Entity Name", 
                placeholder="e.g., EMPLOYEE_ID",
                key="custom_entity_name"
            )
        with col2:
            new_entity_regex = st.text_input(
                "Regex Pattern", 
                placeholder="e.g., \\b[A-Z]{3}\\d{4}\\b",
                key="custom_entity_regex"
            )
        
        # Add Custom Entity Button
        if st.button("Add Custom Entity"):
            if new_entity_name and new_entity_regex:
                self.add_custom_entity(new_entity_name, new_entity_regex)
        
        # Display Existing Custom Entities
        st.subheader("Current Custom Entities")
        if st.session_state.custom_entities:
            # Create a dataframe to display custom entities
            entities_display = []
            for name, pattern in st.session_state.custom_entities.items():
                entities_display.append({
                    "Entity Name": name,
                    "Regex Pattern": pattern
                })
            
            st.dataframe(entities_display)
            
            # Option to remove custom entities
            if st.button("Clear All Custom Entities"):
                st.session_state.custom_entities.clear()
                st.success("All custom entities have been removed.")
        else:
            st.info("No custom entities defined.")

    def render_app(self):
        """Render Streamlit Application"""
        st.title("PII-Safe RAI- RAG Chat Application")
        
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Input Document", 
            "Masked Document", 
            "RAG Chat", 
            "Original Document",
            "Custom Entities"
        ])
        
        with tab1:
            st.header("Document Input")
            # Comprehensive Text Input Area
            input_text = st.text_area(
                "Paste your document here:", 
                height=400, 
                key="document_input",
                placeholder="Enter your text containing potentially sensitive information..."
            )
            
            # Process and Mask Button
            if st.button("Mask Sensitive Information"):
                if input_text:
                    try:
                        # Process text and mask PII
                        masked_text = self.process_text(input_text)
                        
                        # Provide feedback
                        st.success("PII Masking Complete!")
                    except Exception as e:
                        st.error(f"Error during PII masking: {e}")
                        st.error(traceback.format_exc())
        
        with tab2:
            st.header("Masked Document Information")
            
            # Display Masked Text
            if st.session_state.masked_text:
                st.text_area(
                    "Masked Text:", 
                    value=st.session_state.masked_text, 
                    height=400, 
                    disabled=True
                )
            
            # Display Detected Entities
            st.subheader("Detected Sensitive Entities")
            if st.session_state.detected_entities:
                # Create a more informative display of detected entities
                entities_display = []
                for entity in st.session_state.detected_entities:
                    entities_display.append({
                        "Type": entity['entity_type'],
                        "Original Value": entity['original_value'],
                        "Masked Value": entity['masked_value']
                    })
                
                st.dataframe(entities_display)
            else:
                st.info("No sensitive entities detected.")
        
        with tab3:
            st.header("RAG Chat with Masked Document")
            # Chat Input
            query = st.text_input("Ask a question about the document:")
            
            if query and st.session_state.index:
                # Perform RAG Query
                response = self.query_rag(query)
                st.write("Response:", response)
        
        with tab4:
            st.header("Original Document")
            if st.session_state.original_text:
                st.text_area(
                    "Original Text:", 
                    value=st.session_state.original_text, 
                    height=400, 
                    disabled=True
                )
            else:
                st.info("No original document loaded.")
        with tab5:
            # New tab for custom entity configuration
            st.sidebar.write("EMPLOYEE_ID   -  \\b[A-Z]{3}\\d{4}\\b")
            self.render_custom_entity_tab()

def main():
    app = PIISafeRAGApplication()
    app.render_app()

if __name__ == "__main__":
    main()