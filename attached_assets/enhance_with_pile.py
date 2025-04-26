"""
GAKR Chatbot - Enhanced Training with The Pile Dataset
This script uses data from The Pile to enhance the GAKR chatbot's training.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional

# Import our downloader
import pile_downloader

# Import our training module
from enhanced_training import EnhancedDataProcessor, SimpleTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhance_with_pile')

def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from a JSON file.
    
    Args:
        data_path: Path to the training data file
        
    Returns:
        List of training examples
    """
    if not os.path.exists(data_path):
        logger.error(f"Training data file {data_path} not found")
        return []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Loaded {len(data)} training examples from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return []

def enhance_model_with_pile(existing_model_path: str = "./models/trained/simple_model.pkl", 
                           output_model_path: str = "./models/trained/enhanced_pile_model.pkl") -> bool:
    """
    Enhance the existing model with data from The Pile.
    
    Args:
        existing_model_path: Path to the existing trained model
        output_model_path: Path to save the enhanced model
        
    Returns:
        True if enhancement was successful, False otherwise
    """
    try:
        # Step 1: Download and process data from The Pile
        logger.info("Downloading and processing data from The Pile")
        training_data_path = pile_downloader.download_and_process_pile()
        
        if not training_data_path:
            logger.error("Failed to obtain training data from The Pile")
            return False
        
        # Step 2: Create a data processor instance
        logger.info("Creating data processor")
        processed_data_path = "pile_processed_data.json"
        data_processor = EnhancedDataProcessor(output_path=processed_data_path)
        
        # Step 3: Load the training data
        logger.info(f"Loading training data from {training_data_path}")
        training_data = load_training_data(training_data_path)
        
        if not training_data:
            logger.error("No training data available")
            return False
        
        # Step 4: Process the data
        logger.info("Processing the training data")
        for example in training_data:
            # Add each example to the data processor
            data_processor.add_example(example.get("input", ""), example.get("output", ""))
        
        # Process the data
        processed_data = data_processor.process_data(save_results=True)
        
        # Step 5: Train the model
        logger.info("Training the enhanced model")
        trainer = SimpleTrainer(data_processor)
        
        # Create directory for model if it doesn't exist
        model_dir = os.path.dirname(output_model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Train and save the model
        trainer.train(model_path=output_model_path)
        
        logger.info(f"Enhanced model trained and saved to {output_model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to enhance model with Pile data: {e}")
        return False

def main():
    """
    Main function to enhance the model with The Pile.
    """
    parser = argparse.ArgumentParser(description="Enhance GAKR AI model with data from The Pile")
    parser.add_argument("--input-model", default="./models/trained/simple_model.pkl", help="Path to the existing model")
    parser.add_argument("--output-model", default="./models/trained/enhanced_pile_model.pkl", help="Path to save the enhanced model")
    
    args = parser.parse_args()
    
    logger.info("Starting enhancement of GAKR AI model with The Pile data")
    success = enhance_model_with_pile(args.input_model, args.output_model)
    
    if success:
        logger.info("Model enhancement completed successfully")
        print(f"SUCCESS: Model enhanced with The Pile data and saved to {args.output_model}")
    else:
        logger.error("Model enhancement failed")
        print("ERROR: Failed to enhance model with The Pile data")

if __name__ == "__main__":
    main()