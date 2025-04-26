import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for training language models"""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 128):
        """
        Initialize dataset with texts
        
        Args:
            texts: List of text samples
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert to appropriate format
        item = {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
        }
        
        return item

class ConversationDataset(Dataset):
    """Dataset for conversation data with prompt-response pairs"""
    
    def __init__(self, 
                 conversation_pairs: List[Dict[str, str]], 
                 tokenizer, 
                 max_length: int = 128):
        """
        Initialize dataset with conversation pairs
        
        Args:
            conversation_pairs: List of dictionaries with 'prompt' and 'response' keys
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.conversation_pairs = conversation_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversation_pairs)
    
    def __getitem__(self, idx):
        pair = self.conversation_pairs[idx]
        prompt = pair['prompt']
        response = pair['response']
        
        # Format as a single text with appropriate markers
        full_text = f"User: {prompt}\nGAKR: {response}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels are the same as input_ids for causal LM training
        item = {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().clone()
        }
        
        return item

class ModelTrainer:
    """Handles model training and fine-tuning for GAKR"""
    
    def __init__(self, model_path: str, output_dir: str = "./models/trained"):
        """
        Initialize the model trainer
        
        Args:
            model_path: Path to the base model
            output_dir: Directory to save trained models
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model for training: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model for training: {e}")
            raise
    
    def train_on_texts(self, 
                     texts: List[str], 
                     batch_size: int = 4,
                     epochs: int = 3,
                     learning_rate: float = 5e-5) -> str:
        """
        Train the model on a list of texts
        
        Args:
            texts: List of text samples
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Path to the saved model
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create dataset
        dataset = TextDataset(texts, self.tokenizer)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Not using masked language modeling
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            learning_rate=learning_rate,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train model
        logger.info("Starting model training")
        trainer.train()
        
        # Save the trained model
        save_path = os.path.join(self.output_dir, "final")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model trained and saved to {save_path}")
        return save_path
    
    def train_on_conversations(self,
                             conversation_pairs: List[Dict[str, str]],
                             batch_size: int = 4,
                             epochs: int = 3,
                             learning_rate: float = 5e-5) -> str:
        """
        Train the model on conversation pairs
        
        Args:
            conversation_pairs: List of dictionaries with 'prompt' and 'response' keys
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Path to the saved model
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create dataset
        dataset = ConversationDataset(conversation_pairs, self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            learning_rate=learning_rate,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train model
        logger.info("Starting conversation model training")
        trainer.train()
        
        # Save the trained model
        save_path = os.path.join(self.output_dir, "conversation_model")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Conversation model trained and saved to {save_path}")
        return save_path
    
    def auto_train(self, 
                 data_path: str, 
                 data_format: str = "text",
                 **kwargs) -> str:
        """
        Automatically train model based on data in a file
        
        Args:
            data_path: Path to data file (json or txt)
            data_format: Format of data ('text' or 'conversation')
            kwargs: Additional training parameters
            
        Returns:
            Path to the saved model
        """
        # Load data from file
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.endswith(".json"):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith(".txt"):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Unsupported data file format. Use .json or .txt")
        
        # Train based on data format
        if data_format == "text":
            if isinstance(data, list):
                texts = data
            elif isinstance(data, dict) and "texts" in data:
                texts = data["texts"]
            else:
                raise ValueError("Invalid data format for text training")
                
            return self.train_on_texts(
                texts=texts,
                batch_size=kwargs.get("batch_size", 4),
                epochs=kwargs.get("epochs", 3),
                learning_rate=kwargs.get("learning_rate", 5e-5)
            )
            
        elif data_format == "conversation":
            if isinstance(data, list):
                if all(isinstance(item, dict) and "prompt" in item and "response" in item for item in data):
                    conversation_pairs = data
                else:
                    raise ValueError("Conversation data must contain 'prompt' and 'response' keys")
            elif isinstance(data, dict) and "conversations" in data:
                conversation_pairs = data["conversations"]
            else:
                raise ValueError("Invalid data format for conversation training")
                
            return self.train_on_conversations(
                conversation_pairs=conversation_pairs,
                batch_size=kwargs.get("batch_size", 4),
                epochs=kwargs.get("epochs", 3),
                learning_rate=kwargs.get("learning_rate", 5e-5)
            )
            
        else:
            raise ValueError(f"Unsupported data format: {data_format}. Use 'text' or 'conversation'")
