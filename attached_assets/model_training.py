"""
GAKR Chatbot - Model Training Module
This module provides functionality to fine-tune a pre-trained language model
for the GAKR chatbot using custom training data.
"""

import os
import json
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    """Dataset for training the GAKR chatbot model."""
    
    def __init__(self, inputs, outputs, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            inputs: List of input texts (user messages)
            outputs: List of output texts (bot responses)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        
        # Format as a conversation
        full_text = f"User: {input_text}\nGAKR: {output_text}"
        
        # Encode the full conversation
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, the labels are the same as the input_ids
        encodings["labels"] = encodings["input_ids"].clone()
        
        # Remove the batch dimension
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["labels"].squeeze()
        }


def load_training_data(data_path):
    """
    Load training data from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing training data
        
    Returns:
        Tuple of (inputs, outputs)
    """
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        
        inputs = [item["input"] for item in data]
        outputs = [item["output"] for item in data]
        
        return inputs, outputs
    
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        # Fallback to default minimal dataset
        return [
            "What is your name?",
            "Tell me a joke.",
            "How are you?",
            "What can you do?"
        ], [
            "I am GAKR, an AI chatbot built to process and analyze text.",
            "Why don't scientists trust atoms? Because they make up everything!",
            "I'm functioning well, thank you for asking! How can I assist you today?",
            "I can analyze sentiment, answer questions, and have conversations with you."
        ]


def train_model(
    model_name="gpt2-small",  # Use smaller model to fit in Replit
    data_path="training_data.json",
    output_dir="./models/trained",
    num_train_epochs=3,
    learning_rate=5e-5,
    batch_size=4,
    max_seq_length=128
):
    """
    Train a language model for the GAKR chatbot.
    
    Args:
        model_name: Name or path of the pre-trained model
        data_path: Path to the training data file
        output_dir: Directory to save the trained model
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        max_seq_length: Maximum sequence length
    """
    logger.info(f"Starting model training process with {model_name}...")
    
    try:
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        
        # Use GPT-2 as a fallback since it's smaller and widely compatible
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Error loading {model_name}: {e}. Falling back to GPT-2...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load training data
        logger.info("Loading training data...")
        inputs, outputs = load_training_data(data_path)
        logger.info(f"Loaded {len(inputs)} training examples")
        
        # Create dataset
        train_dataset = ConversationDataset(inputs, outputs, tokenizer, max_seq_length)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        
        # Train the model
        logger.info("Training model...")
        trainer.train()
        
        # Save the trained model
        logger.info(f"Saving trained model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Model training completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False


def create_default_training_data(output_path="training_data.json"):
    """
    Create a default training data file with examples.
    
    Args:
        output_path: Path to save the training data
    """
    try:
        # Create training examples from our knowledge base and patterns
        data = [
            {"input": "What is your name?", "output": "I am GAKR, an AI chatbot built to process and analyze text."},
            {"input": "Tell me a joke.", "output": "Why don't scientists trust atoms? Because they make up everything!"},
            {"input": "How are you?", "output": "I'm functioning well, thank you for asking! How can I assist you today?"},
            {"input": "What can you do?", "output": "I can analyze sentiment, answer questions, and have conversations with you."},
            {"input": "What is Python?", "output": "Python is an interpreted, high-level, general-purpose programming language known for its readability and versatility."},
            {"input": "What is AI?", "output": "AI (Artificial Intelligence) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."},
            {"input": "Tell me about GAKR.", "output": "GAKR is an AI chatbot built to process and analyze text using pre-trained models without requiring external API dependencies."},
            {"input": "What is HTML?", "output": "HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser."},
            {"input": "What does BTech mean?", "output": "BTech (Bachelor of Technology) is an undergraduate academic degree conferred after completion of a four-year program in engineering or technology."},
            {"input": "What is a library?", "output": "A library is a collection of sources of information and similar resources, made accessible to a defined community for reference or borrowing."}
        ]
        
        # Write to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created default training data file at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating training data file: {e}")
        return False


if __name__ == "__main__":
    # Create default training data if it doesn't exist
    if not os.path.exists("training_data.json"):
        create_default_training_data()
    
    # Train model (use a small model to fit in Replit environment)
    train_model(model_name="distilgpt2", num_train_epochs=1)