"""
GAKR Chatbot - Data Merger Script
This script merges all training datasets into a comprehensive training set.
"""

import json
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of data items
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the data to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} items to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def merge_datasets(input_files: List[str], output_file: str) -> bool:
    """
    Merge multiple datasets into one.
    
    Args:
        input_files: List of input file paths
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    merged_data = []
    seen_inputs = set()
    
    for file_path in input_files:
        data = load_json_data(file_path)
        
        for item in data:
            input_text = item["input"].lower()
            if input_text not in seen_inputs:
                merged_data.append(item)
                seen_inputs.add(input_text)
            else:
                logger.debug(f"Skipping duplicate input: {input_text}")
    
    logger.info(f"Merged data contains {len(merged_data)} items")
    return save_json_data(merged_data, output_file)

def main():
    """
    Main function to merge all datasets.
    """
    logger.info("Starting dataset merger")
    
    # List all input files
    input_files = [
        "training_data.json",              # Original training data
        "enhanced_training_data.json",     # Enhanced data with Cornell Movie dataset
        "ai_capabilities_data.json",       # AI capabilities data
        "ai_technical_data_fixed.json",    # AI technical data (fixed version)
        "datasets_training_data.json"      # Training datasets information
    ]
    
    # Filter existing files
    existing_files = [f for f in input_files if os.path.exists(f)]
    logger.info(f"Found {len(existing_files)} existing datasets: {existing_files}")
    
    if not existing_files:
        logger.error("No datasets found to merge")
        return False
    
    # Merge datasets
    result = merge_datasets(existing_files, "comprehensive_training_data.json")
    
    if result:
        logger.info("Successfully merged all datasets")
        return True
    else:
        logger.error("Failed to merge datasets")
        return False

if __name__ == "__main__":
    main()