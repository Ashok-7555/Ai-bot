"""
GAKR Chatbot - Enhanced Model Training CLI
This script provides a command-line interface for training the enhanced GAKR chatbot model.
"""

import os
import sys
import logging
import argparse
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modules
import dataset_downloader
from enhanced_training import EnhancedDataProcessor, SimpleTrainer


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train the enhanced GAKR chatbot model.")
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["persona-chat", "dailydialog", "cornell-movie"],
        help="Datasets to use for training"
    )
    
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing training data in addition to downloaded datasets"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./models/trained",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--model-name",
        default="simple_model.pkl",
        help="Name of the trained model file"
    )
    
    return parser.parse_args()


def train_model(dataset_paths: List[str], output_dir: str, model_name: str) -> bool:
    """
    Train the enhanced model using the specified datasets.
    
    Args:
        dataset_paths: List of paths to datasets to use for training
        output_dir: Directory to save the trained model
        model_name: Name of the trained model file
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Create data processor
        processor = EnhancedDataProcessor()
        
        # Load all datasets
        for path in dataset_paths:
            logger.info(f"Loading dataset from {path}")
            if path.endswith(".json"):
                processor.load_custom_data(path)
            else:
                logger.warning(f"Unsupported dataset format: {path}")
        
        # Process data
        processor.process_data()
        
        # Train model
        model_path = os.path.join(output_dir, model_name)
        trainer = SimpleTrainer(processor)
        trainer.train(model_path=model_path)
        
        logger.info(f"Model trained successfully and saved to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False


def main():
    """
    Main function to train the enhanced GAKR model.
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download datasets
    dataset_paths = []
    for dataset_name in args.datasets:
        logger.info(f"Downloading dataset: {dataset_name}")
        path = dataset_downloader.download_dataset(dataset_name)
        if path:
            dataset_paths.append(path)
    
    if not dataset_paths:
        logger.error("No datasets were downloaded successfully. Aborting training.")
        return 1
    
    # Add existing training data if requested
    if args.use_existing and os.path.exists("training_data.json"):
        logger.info("Adding existing training data")
        dataset_paths.append("training_data.json")
    
    # Train model
    success = train_model(dataset_paths, args.output_dir, args.model_name)
    
    if success:
        logger.info("Training completed successfully!")
        return 0
    else:
        logger.error("Training failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())