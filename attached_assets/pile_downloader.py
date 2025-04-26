"""
GAKR Chatbot - The Pile Dataset Downloader
This module downloads and processes a small sample of data from The Pile dataset
using direct HTTP requests and minimal dependencies.
"""

import os
import json
import random
import logging
import urllib.request
import urllib.error
import time
import re
import gzip
import shutil
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pile_downloader')

# Constants
PILE_SAMPLE_URL = "https://huggingface.co/datasets/NeelNanda/pile-10k/resolve/main/val.jsonl"
SAMPLE_SIZE = 1000  # Number of samples to include in our dataset
OUTPUT_DIR = "./datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pile_samples.json")
CONVERSATION_FILE = os.path.join(OUTPUT_DIR, "pile_conversations.json") 

def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url} to {output_path}")
        
        # Attempt to download with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                logger.info(f"Download successful")
                return True
            except urllib.error.URLError as e:
                logger.warning(f"Download attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
        
        return False
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        return False

def extract_gzip(gzip_path: str, extract_path: str) -> bool:
    """
    Extract a gzip file.
    
    Args:
        gzip_path: Path to the gzip file
        extract_path: Path to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting {gzip_path} to {extract_path}")
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Extraction successful")
        return True
    except Exception as e:
        logger.error(f"Failed to extract file: {e}")
        return False

def process_pile_sample(jsonl_path: str, output_path: str, sample_size: int = SAMPLE_SIZE) -> bool:
    """
    Process The Pile sample file and save a smaller random subset.
    
    Args:
        jsonl_path: Path to the JSONL file
        output_path: Path to save the processed samples
        sample_size: Number of samples to collect
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing Pile sample file: {jsonl_path}")
        
        # Read all lines from the JSONL file
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Take a random sample if there are more entries than sample_size
        if len(lines) > sample_size:
            logger.info(f"Selecting {sample_size} random samples from {len(lines)} entries")
            lines = random.sample(lines, sample_size)
        
        # Parse JSON objects from the selected lines
        samples = []
        for line in lines:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
        
        # Save processed samples
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(samples, out_file, indent=2)
        
        logger.info(f"Saved {len(samples)} processed samples to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to process Pile samples: {e}")
        return False

def create_conversation_pairs(samples_path: str, output_path: str, min_length: int = 50, max_length: int = 2000) -> bool:
    """
    Create conversation pairs from The Pile samples.
    
    Args:
        samples_path: Path to the samples JSON file
        output_path: Path to save the conversation pairs
        min_length: Minimum text length to consider
        max_length: Maximum text length to consider
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Creating conversation pairs from {samples_path}")
        
        with open(samples_path, 'r', encoding='utf-8') as file:
            samples = json.load(file)
        
        conversation_pairs = []
        
        for sample in samples:
            text = sample.get('text', '')
            
            # Skip very short or very long texts
            if len(text) < min_length or len(text) > max_length:
                continue
            
            # Try to split into paragraphs to create Q&A pairs
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            
            if len(paragraphs) >= 2:
                # Create a Q&A pair from consecutive paragraphs
                for i in range(len(paragraphs) - 1):
                    # Convert the first paragraph into a question if it's not already
                    p1 = paragraphs[i]
                    p2 = paragraphs[i+1]
                    
                    # Skip if either paragraph is too short
                    if len(p1) < 20 or len(p2) < 30:
                        continue
                    
                    # If p1 doesn't end with a question mark, try to formulate a question
                    question = p1
                    if not p1.strip().endswith('?'):
                        # Extract key terms to create a question
                        words = re.findall(r'\b\w+\b', p1.lower())
                        important_words = [w for w in words if len(w) > 4 and w not in ['about', 'these', 'those', 'their', 'would', 'could', 'should']]
                        
                        if important_words:
                            key_term = random.choice(important_words)
                            question = f"Can you tell me more about {key_term} mentioned in the text?"
                    
                    conversation_pairs.append({
                        "input": question,
                        "response": p2,
                        "source": sample.get('meta', {}).get('pile_set_name', 'The Pile')
                    })
                    
                    # Limit to one pair per sample to avoid duplicate information
                    break
        
        # Save conversation pairs
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(conversation_pairs, out_file, indent=2)
        
        logger.info(f"Created {len(conversation_pairs)} conversation pairs saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create conversation pairs: {e}")
        return False

def convert_to_training_format(conversations_path: str, output_path: str = "pile_training_data.json") -> bool:
    """
    Convert conversation pairs to GAKR training format.
    
    Args:
        conversations_path: Path to the conversation pairs file
        output_path: Path to save the training data
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        logger.info(f"Converting conversation pairs to training format")
        
        with open(conversations_path, 'r', encoding='utf-8') as file:
            conversations = json.load(file)
        
        training_data = []
        
        for item in conversations:
            input_text = item.get('input', '')
            response = item.get('response', '')
            source = item.get('source', 'The Pile')
            
            # Skip if either input or response is missing
            if not input_text or not response:
                continue
            
            # Create training example
            training_example = {
                "input": input_text,
                "output": response,
                "meta": {
                    "source": source,
                    "category": "general_knowledge",
                    "quality": "medium"
                }
            }
            
            training_data.append(training_example)
        
        # Save training data
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(training_data, out_file, indent=2)
        
        logger.info(f"Converted {len(training_data)} examples to training format, saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert to training format: {e}")
        return False

def merge_with_existing_data(new_data_path: str, existing_data_path: str, output_path: str = "merged_training_data.json") -> bool:
    """
    Merge new training data with existing training data.
    
    Args:
        new_data_path: Path to the new training data
        existing_data_path: Path to the existing training data
        output_path: Path to save the merged data
        
    Returns:
        True if merging was successful, False otherwise
    """
    try:
        logger.info(f"Merging {new_data_path} with {existing_data_path}")
        
        # Load new data
        with open(new_data_path, 'r', encoding='utf-8') as file:
            new_data = json.load(file)
        
        # Load existing data if it exists
        if os.path.exists(existing_data_path):
            with open(existing_data_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        else:
            logger.warning(f"Existing data file {existing_data_path} not found, using only new data")
            existing_data = []
        
        # Merge data
        merged_data = existing_data + new_data
        
        # Save merged data
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(merged_data, out_file, indent=2)
        
        logger.info(f"Merged data saved to {output_path}: {len(existing_data)} existing + {len(new_data)} new = {len(merged_data)} total examples")
        return True
    except Exception as e:
        logger.error(f"Failed to merge data: {e}")
        return False

def download_and_process_pile() -> Optional[str]:
    """
    Download and process a sample from The Pile dataset.
    
    Returns:
        Path to the processed training data, or None if processing failed
    """
    # Create directories
    ensure_directory(OUTPUT_DIR)
    
    # Temporary filenames
    jsonl_file = os.path.join(OUTPUT_DIR, "mini_pile.jsonl")
    
    # Download JSONL file
    if not download_file(PILE_SAMPLE_URL, jsonl_file):
        logger.error("Failed to download The Pile sample")
        return None
    
    # Process the downloaded JSONL file
    try:
        # Process each line as a separate JSON object
        samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict):
                        samples.append(item)
                    else:
                        # Try to convert to the right format
                        sample = {"text": str(item)}
                        samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
            
        # Save to our expected output file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(samples, outfile, indent=2)
            
        logger.info(f"Processed {len(samples)} samples from The Pile")
        
        if not samples:
            logger.error("No valid samples were found in the downloaded file")
            return None
    except Exception as e:
        logger.error(f"Failed to process the downloaded JSONL file: {e}")
        return None
    
    # Create conversation pairs
    if not create_conversation_pairs(OUTPUT_FILE, CONVERSATION_FILE):
        logger.error("Failed to create conversation pairs")
        return None
    
    # Convert to training format
    training_file = os.path.join(OUTPUT_DIR, "pile_training_data.json")
    if not convert_to_training_format(CONVERSATION_FILE, training_file):
        logger.error("Failed to convert to training format")
        return None
    
    # Merge with existing data if available
    existing_data = "training_data.json"
    merged_file = "merged_training_data.json"
    if os.path.exists(existing_data):
        if not merge_with_existing_data(training_file, existing_data, merged_file):
            logger.error("Failed to merge with existing data")
            return training_file  # Return the new data only
    else:
        # No existing data, just copy the new data
        shutil.copy(training_file, merged_file)
    
    # Cleanup temporary files
    if os.path.exists(jsonl_file):
        os.remove(jsonl_file)
        logger.info(f"Removed temporary file: {jsonl_file}")
    
    return merged_file

def main():
    """
    Main function to download and process The Pile dataset.
    """
    logger.info("Starting download and processing of The Pile dataset")
    output_path = download_and_process_pile()
    
    if output_path:
        logger.info(f"Successfully processed The Pile data, saved to {output_path}")
        print(f"SUCCESS: The Pile data saved to {output_path}")
    else:
        logger.error("Failed to process The Pile data")
        print("ERROR: Failed to process The Pile data")

if __name__ == "__main__":
    main()