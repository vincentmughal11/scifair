import os
import csv
import time
import requests
import argparse
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("humanization.log"),
        logging.StreamHandler()
    ]
)

API_BASE_URL = "https://humanize.undetectable.ai"
INPUT_FILE = "./clean/ai_to_humanize_limited.csv"
OUTPUT_FILE = "./clean/ai_humanized_samples.csv"

class UndetectableAIHumanizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        # Default settings
        self.readability = "University"
        self.purpose = "Essay"
        self.strength = "Balanced"

    def submit_document(self, text: str) -> Optional[str]:
        """Submits a document for humanization and returns the document ID."""
        url = f"{API_BASE_URL}/submit"
        payload = {
            "content": text,
            "readability": self.readability,
            "purpose": self.purpose,
            "strength": self.strength,
            "model": "v2",
            "document_type": "Text"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # The API returns the document ID in the response, usually as 'id' or similar.
            # Based on typical patterns, let's assume it returns the ID directly or in a field.
            # Looking at the openapi.json, the response schema is empty {}, which is unhelpful.
            # I will assume standard behavior and log the response to debug if needed.
            # Wait, looking at similar APIs, it might return just the ID or an object.
            # Let's try to parse 'id' from the response.
            if 'id' in data:
                return data['id']
            else:
                # Fallback: sometimes these APIs return the ID as a string if the schema is loose
                # Or maybe it's in a 'data' field.
                logging.info(f"Submission response: {data}")
                return data.get('id') 
        except requests.exceptions.RequestException as e:
            logging.error(f"Error submitting document: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return None

    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the document status/content."""
        url = f"{API_BASE_URL}/document"
        payload = {"id": doc_id}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error retrieving document {doc_id}: {e}")
            return None

    def humanize_text(self, text: str) -> Optional[str]:
        """Orchestrates the submission and polling for a single text."""
        if not text or len(text.strip()) < 10:
            logging.warning("Skipping empty or too short text.")
            return None

        doc_id = self.submit_document(text)
        if not doc_id:
            return None
        
        logging.info(f"Document submitted. ID: {doc_id}. Waiting for processing...")
        
        # Poll for completion
        max_retries = 30  # 30 * 10s = 5 minutes max wait
        for _ in range(max_retries):
            time.sleep(10) # Wait 10 seconds between checks
            doc_data = self.get_document_status(doc_id)
            
            if not doc_data:
                continue
                
            status = doc_data.get('status')
            
            if status == 'done':
                # Success!
                # The output text is likely in 'output' or 'humanized_content'
                # Let's inspect the likely fields based on the 'RetrieveDocumentRequest' response
                # The schema didn't specify, but 'output' is common.
                # Let's look for 'output' or 'content' in the response.
                # If the status is done, the humanized text should be there.
                return doc_data.get('output') or doc_data.get('content')
            elif status == 'failed':
                logging.error(f"Humanization failed for document {doc_id}.")
                return None
            
            logging.info(f"Document {doc_id} status: {status}")
            
        logging.error(f"Timeout waiting for document {doc_id}.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Humanize AI text samples.")
    parser.add_argument("--test", action="store_true", help="Run a test with a single sample.")
    args = parser.parse_args()

    api_key = os.environ.get("UNDETECTABLE_AI_API_KEY")
    if not api_key:
        # Try to ask user for input if not set (for local running)
        print("UNDETECTABLE_AI_API_KEY environment variable not set.")
        # For this script, we'll just exit if not set to avoid hanging in non-interactive modes unexpectedly
        return

    humanizer = UndetectableAIHumanizer(api_key)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load existing progress to avoid duplicates
    processed_texts = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use original text as key if available, or just track count?
                # Tracking by text content is safer for restarts.
                # But the output CSV might not have the original text if we only save the new one.
                # Let's assume we append to the file.
                pass
    
    # We'll append to the file, so open in 'a' mode later.
    # But first, let's read the input.
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            samples = list(reader)
    except FileNotFoundError:
        logging.error(f"Input file {INPUT_FILE} not found.")
        return

    logging.info(f"Found {len(samples)} samples.")

    if args.test:
        logging.info("Running in TEST mode (1 sample).")
        samples = samples[:1]

    # Open output file for appending
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['original_text', 'humanized_text', 'original_label', 'new_label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for i, sample in enumerate(samples):
            original_text = sample.get('text')
            label = sample.get('label')
            
            logging.info(f"Processing sample {i+1}/{len(samples)}...")
            
            humanized = humanizer.humanize_text(original_text)
            
            if humanized:
                writer.writerow({
                    'original_text': original_text,
                    'humanized_text': humanized,
                    'original_label': label,
                    'new_label': 'AI_HUMANIZED'
                })
                f.flush() # Ensure data is written
                logging.info("Sample saved.")
            else:
                logging.error("Failed to humanize sample.")
            
            # Rate limiting / politeness
            if not args.test:
                time.sleep(2) 

if __name__ == "__main__":
    main()
