"""
GAKR Chatbot - Dataset Downloader
This module provides functionality to download and prepare datasets for training the GAKR chatbot.
"""

import os
import json
import logging
import requests
import random
import shutil
import time
import zipfile
import re
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset sources
DATASET_SOURCES = {
    "persona-chat": {
        "url": "https://raw.githubusercontent.com/microsoft/PersonalityChat/master/Datasets/train_both_revised.tsv",
        "description": "Microsoft Personality Chat dataset with casual and professional responses",
        "format": "tsv"
    },
    "dailydialog": {
        "url": "https://raw.githubusercontent.com/HLTCHKUST/sentiment-empathetic-nlg/master/data/dailydialog/dialogues_train.txt",
        "description": "DailyDialog dataset with conversations on common topics",
        "format": "txt"
    },
    "cornell-movie": {
        "description": "Cornell Movie Dialogs Corpus with conversations from movie scripts",
        "local": True
    }
}

# Directory for datasets
DATASET_DIR = "./datasets"


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
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {url} to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted {zip_path} to {extract_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False


def create_cornell_movie_dataset() -> str:
    """
    Create a simple version of the Cornell Movie Dialogs dataset.
    
    Returns:
        Path to the created dataset
    """
    output_path = os.path.join(DATASET_DIR, "cornell-movie.json")
    
    # Create a simplified version with a few example dialogs
    dialogs = [
        {"input": "Hello.", "output": "Hi, how are you doing?"},
        {"input": "What's up?", "output": "Not much. What's new with you?"},
        {"input": "I'm good, thanks.", "output": "Glad to hear that."},
        {"input": "How's your day?", "output": "It's been great. Thanks for asking."},
        {"input": "What do you think about movies?", "output": "I enjoy watching them, especially classics."},
        {"input": "Do you like music?", "output": "Yes, music is a universal language that connects people."},
        {"input": "What's your favorite food?", "output": "If I could eat, I'd probably enjoy trying different cuisines."},
        {"input": "Can you help me?", "output": "Of course, what do you need help with?"},
        {"input": "Tell me about yourself.", "output": "I'm an AI assistant designed to have natural conversations and provide information."},
        {"input": "Thanks for your help.", "output": "You're welcome! I'm happy to assist anytime."}
    ]
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(dialogs, f, indent=2)
    
    logger.info(f"Created Cornell Movie dataset at {output_path}")
    return output_path


def convert_persona_chat(input_path: str) -> str:
    """
    Convert Persona-Chat dataset to GAKR format.
    
    Args:
        input_path: Path to the input TSV file
        
    Returns:
        Path to the converted dataset
    """
    output_path = os.path.join(DATASET_DIR, "persona-chat.json")
    
    try:
        dialogs = []
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    input_text = parts[0].strip()
                    output_text = parts[1].strip()
                    
                    if input_text and output_text:
                        dialogs.append({
                            "input": input_text,
                            "output": output_text
                        })
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(dialogs, f, indent=2)
        
        logger.info(f"Converted Persona-Chat dataset to {output_path} with {len(dialogs)} dialogs")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting Persona-Chat dataset: {e}")
        return None


def convert_dailydialog(input_path: str) -> str:
    """
    Convert DailyDialog dataset to GAKR format.
    
    Args:
        input_path: Path to the input text file
        
    Returns:
        Path to the converted dataset
    """
    output_path = os.path.join(DATASET_DIR, "dailydialog.json")
    
    try:
        dialogs = []
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
            dialog_blocks = content.strip().split("\n\n")
            
            for block in dialog_blocks:
                lines = block.strip().split("\n")
                for i in range(len(lines) - 1):
                    input_text = lines[i].strip()
                    output_text = lines[i + 1].strip()
                    
                    if input_text and output_text:
                        dialogs.append({
                            "input": input_text,
                            "output": output_text
                        })
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(dialogs, f, indent=2)
        
        logger.info(f"Converted DailyDialog dataset to {output_path} with {len(dialogs)} dialogs")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting DailyDialog dataset: {e}")
        return None


def merge_datasets(dataset_paths: List[str], output_path: str = "merged_training_data.json") -> str:
    """
    Merge multiple datasets into one.
    
    Args:
        dataset_paths: List of paths to datasets to merge
        output_path: Path to save the merged dataset
        
    Returns:
        Path to the merged dataset
    """
    try:
        dialogs = []
        
        for path in dataset_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dialogs.extend(data)
                logger.info(f"Added {len(data)} dialogs from {path}")
            except Exception as e:
                logger.error(f"Error loading dataset {path}: {e}")
        
        # Remove duplicates
        unique_dialogs = []
        seen_inputs = set()
        
        for dialog in dialogs:
            input_text = dialog["input"].lower()
            if input_text not in seen_inputs:
                unique_dialogs.append(dialog)
                seen_inputs.add(input_text)
        
        logger.info(f"Removed {len(dialogs) - len(unique_dialogs)} duplicate dialogs")
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(unique_dialogs, f, indent=2)
        
        logger.info(f"Merged {len(dataset_paths)} datasets into {output_path} with {len(unique_dialogs)} dialogs")
        return output_path
    
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        return None


def download_dataset(dataset_name: str) -> Optional[str]:
    """
    Download a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to download
        
    Returns:
        Path to the downloaded and processed dataset, or None if download failed
    """
    ensure_directory(DATASET_DIR)
    
    if dataset_name not in DATASET_SOURCES:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None
    
    dataset_info = DATASET_SOURCES[dataset_name]
    
    # Check if dataset is local only
    if dataset_info.get("local", False):
        if dataset_name == "cornell-movie":
            return create_cornell_movie_dataset()
        else:
            logger.error(f"Local dataset {dataset_name} not implemented")
            return None
    
    # Download the dataset
    download_path = os.path.join(DATASET_DIR, f"{dataset_name}.{dataset_info['format']}")
    success = download_file(dataset_info["url"], download_path)
    
    if not success:
        return None
    
    # Convert to GAKR format
    if dataset_name == "persona-chat":
        return convert_persona_chat(download_path)
    elif dataset_name == "dailydialog":
        return convert_dailydialog(download_path)
    else:
        logger.error(f"Converter for {dataset_name} not implemented")
        return None


def download_all_datasets() -> List[str]:
    """
    Download all available datasets.
    
    Returns:
        List of paths to downloaded datasets
    """
    dataset_paths = []
    
    for dataset_name in DATASET_SOURCES.keys():
        path = download_dataset(dataset_name)
        if path:
            dataset_paths.append(path)
    
    return dataset_paths


def main():
    """
    Main function to download and prepare datasets.
    """
    logger.info("Starting dataset download")
    
    # Download all datasets
    dataset_paths = download_all_datasets()
    
    # Merge datasets
    if dataset_paths:
        merged_path = merge_datasets(dataset_paths)
        
        if merged_path:
            logger.info(f"Successfully prepared datasets at {merged_path}")
            
            # Also merge with existing training data
            if os.path.exists("training_data.json"):
                logger.info("Merging with existing training data")
                all_paths = dataset_paths + ["training_data.json"]
                final_path = merge_datasets(all_paths, "enhanced_training_data.json")
                
                if final_path:
                    logger.info(f"Created enhanced training data at {final_path}")
    else:
        logger.error("No datasets were downloaded successfully")


if __name__ == "__main__":
    main()