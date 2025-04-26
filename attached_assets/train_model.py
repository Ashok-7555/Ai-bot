#!/usr/bin/env python3
"""
GAKR Chatbot - Model Training CLI
This script provides a command-line interface for training the GAKR chatbot model.
"""

import argparse
import os
import sys
import logging
from model_training import train_model, create_default_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the GAKR model training CLI.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model for the GAKR chatbot."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="distilgpt2", 
        help="Pre-trained model to use as a base (default: distilgpt2)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default="training_data.json", 
        help="Path to training data file (default: training_data.json)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./models/trained", 
        help="Directory to save trained model (default: ./models/trained)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4, 
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate (default: 5e-5)"
    )
    
    parser.add_argument(
        "--create-data", 
        action="store_true", 
        help="Create default training data file if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Create default training data if requested
    if args.create_data and not os.path.exists(args.data):
        logger.info(f"Creating default training data at {args.data}")
        if create_default_training_data(args.data):
            logger.info("Default training data created successfully")
        else:
            logger.error("Failed to create default training data")
            return 1
    
    # Ensure training data exists
    if not os.path.exists(args.data):
        logger.error(f"Training data file {args.data} not found")
        logger.info("You can create a default training data file with --create-data")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Train the model
    logger.info(f"Training model {args.model} with data from {args.data}")
    success = train_model(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        logger.info(f"Model training completed successfully")
        logger.info(f"Trained model saved to {args.output}")
        return 0
    else:
        logger.error("Model training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())