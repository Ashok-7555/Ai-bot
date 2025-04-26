"""
Model loader for the GAKR chatbot.
Handles loading pre-trained models from local storage and Hugging Face.
"""

import json
import logging
import os
from typing import Dict, Tuple, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles loading and caching of NLP models for the chatbot.
    """
    
    def __init__(self):
        """Initialize the model loader with configuration."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model loader initialized with device: {self.device}")
        
        # Load model configuration
        self.config = self._load_model_config()
        
    def _load_model_config(self) -> Dict:
        """
        Load model configuration from JSON file.
        
        Returns:
            Dictionary containing model configuration
        """
        try:
            config_path = os.path.join("models", "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return json.load(f)
            else:
                # Return default configuration if file doesn't exist
                return {
                    "sentiment_analysis": {
                        "model": "distilbert-base-uncased-finetuned-sst-2-english",
                        "fallback": "distilbert-base-uncased-finetuned-sst-2-english"
                    },
                    "text_generation": {
                        "model": "gpt2",
                        "fallback": "distilgpt2"
                    },
                    "question_answering": {
                        "model": "distilbert-base-cased-distilled-squad",
                        "fallback": "distilbert-base-cased-distilled-squad"
                    }
                }
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            # Return default configuration if loading fails
            return {
                "sentiment_analysis": {
                    "model": "distilbert-base-uncased-finetuned-sst-2-english",
                    "fallback": "distilbert-base-uncased-finetuned-sst-2-english"
                },
                "text_generation": {
                    "model": "gpt2",
                    "fallback": "distilgpt2"
                },
                "question_answering": {
                    "model": "distilbert-base-cased-distilled-squad",
                    "fallback": "distilbert-base-cased-distilled-squad"
                }
            }
    
    def load_sentiment_model(self, fallback: bool = False) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load the sentiment analysis model.
        
        Args:
            fallback: Whether to load the fallback model
            
        Returns:
            Tuple of model and tokenizer
        """
        try:
            model_name = self.config["sentiment_analysis"]["fallback" if fallback else "model"]
            logger.info(f"Loading sentiment model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            if not fallback:
                # Try loading fallback model
                return self.load_sentiment_model(fallback=True)
            else:
                # Load minimal DistilBERT model as last resort
                logger.info("Loading minimal DistilBERT sentiment model")
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(self.device)
                return model, tokenizer
    
    def load_text_generation_model(self, fallback: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the text generation model.
        
        Args:
            fallback: Whether to load the fallback model
            
        Returns:
            Tuple of model and tokenizer
        """
        try:
            model_name = self.config["text_generation"]["fallback" if fallback else "model"]
            logger.info(f"Loading text generation model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading text generation model: {e}")
            if not fallback:
                # Try loading fallback model
                return self.load_text_generation_model(fallback=True)
            else:
                # Load minimal GPT-2 model as last resort
                logger.info("Loading minimal GPT-2 model")
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
                model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
                return model, tokenizer
    
    def load_question_answering_model(self, fallback: bool = False) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
        """
        Load the question answering model.
        
        Args:
            fallback: Whether to load the fallback model
            
        Returns:
            Tuple of model and tokenizer
        """
        try:
            model_name = self.config["question_answering"]["fallback" if fallback else "model"]
            logger.info(f"Loading question answering model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading question answering model: {e}")
            if not fallback:
                # Try loading fallback model
                return self.load_question_answering_model(fallback=True)
            else:
                # Load minimal DistilBERT QA model as last resort
                logger.info("Loading minimal DistilBERT QA model")
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(self.device)
                return model, tokenizer
