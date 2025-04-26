import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import kagglehub

from core.utils.text_processor import TextProcessor
from core.utils.spell_checker import SpellChecker

logger = logging.getLogger(__name__)

class NLPEngine:
    """
    Main NLP processing engine for the GAKR AI chatbot.
    Handles loading models, processing inputs, and generating responses.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the NLP engine with specified model
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on (cpu, cuda, etc.)
        """
        self.model_name = model_name or "gpt2"  # Default to GPT-2 if no model specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_processor = TextProcessor()
        self.spell_checker = SpellChecker()
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        self.max_context_length = 1024
        self.model_initialized = False
        
    def initialize_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Check if we need to use Kaggle model
            if self.model_name.startswith("kaggle:"):
                kaggle_model = self.model_name.replace("kaggle:", "")
                logger.info(f"Downloading Kaggle model: {kaggle_model}")
                model_path = kagglehub.model_download(kaggle_model)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Load from Hugging Face
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Move model to specified device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.model_initialized = True
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a smaller model if the requested one fails
            if self.model_name != "gpt2":
                logger.info("Falling back to GPT-2 model")
                self.model_name = "gpt2"
                self.initialize_model()
            else:
                raise
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess the input text before sending to the model
        
        Args:
            text: Raw input text from user
            
        Returns:
            Preprocessed text
        """
        # Correct spelling
        corrected_text = self.spell_checker.correct(text)
        
        # Clean and normalize text
        processed_text = self.text_processor.normalize(corrected_text)
        
        return processed_text
    
    def generate_response(self, 
                          text: str, 
                          conversation_history: List[Dict[str, str]] = None, 
                          max_new_tokens: int = 150) -> Tuple[str, float]:
        """
        Generate a response based on input text and conversation history.
        Falls back to rule-based responses if model generation fails.
        """
        try:
            if not self.model_initialized:
                # Use rule-based response if model isn't initialized
                response = self._get_rule_based_response(text)
                return response, 0.7
            
            # Use the transformer model if available
            processed_text = self.preprocess_input(text)
            response = self._generate_with_model(processed_text, conversation_history, max_new_tokens)
            return response, 0.8
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Fallback to simple response
            return "I'm here to help. What would you like to know?", 0.5
            
    def _get_rule_based_response(self, text: str) -> str:
        """Generate a rule-based response when model is unavailable."""
        text = text.lower()
        if any(word in text for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        elif "help" in text:
            return "I'm here to help. What would you like to know about?"
        elif "thank" in text:
            return "You're welcome! Let me know if you need anything else."
        else:
            return "I understand. Could you tell me more about what you'd like to know?"
            
    def _generate_with_model(self, text: str, history: List = None, max_tokens: int = 150) -> str:
        """Generate response using the transformer model."""
        try:
            # Format conversation history
            context = ""
            if history:
                for turn in history[-3:]:  # Use last 3 turns
                    if isinstance(turn, dict):
                        role = turn.get('role', '')
                        content = turn.get('content', '')
                        if role and content:
                            context += f"{role}: {content}\n"
            
            # Prepare input
            full_input = f"{context}User: {text}\nAssistant:"
            
            # Generate response
            inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            return self._get_rule_based_response(text)
        """
        Generate a response based on input text and conversation history
        
        Args:
            text: User input text
            conversation_history: Previous conversation turns
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (generated response, confidence score)
        """
        if not self.model_initialized:
            self.initialize_model()
            
        # Preprocess input text
        processed_text = self.preprocess_input(text)
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            for turn in conversation_history[-5:]:  # Use the last 5 turns for context
                if "user" in turn:
                    context += f"User: {turn['user']}\n"
                if "assistant" in turn:
                    context += f"GAKR: {turn['assistant']}\n"
        
        # Prepare full input
        full_input = f"{context}User: {processed_text}\nGAKR:"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(full_input, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the new part (the assistant's response)
            response = generated_text.split("GAKR:")[-1].strip()
            
            # Simple confidence score (placeholder)
            confidence = 0.85  # In a real system, this would be calculated
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an issue processing your request. Please try again.", 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self.model_initialized,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
