import os
import logging
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import kagglehub

from core.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    """
    Transformer-based language model implementation
    """
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize a transformer model
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.max_length = 512
    
    def initialize(self) -> None:
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading transformer model: {self.model_name}")
            
            # Handle Kaggle models
            if self.model_name.startswith("kaggle:"):
                kaggle_model = self.model_name.replace("kaggle:", "")
                model_path = kagglehub.model_download(kaggle_model)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Regular HuggingFace models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Transformer model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            # Fall back to a smaller model
            if "gpt2" not in self.model_name.lower():
                logger.info("Falling back to GPT-2 model")
                self.model_name = "gpt2"
                self.initialize()
    
    def generate(self, 
                input_text: str, 
                context: Optional[List[Dict[str, str]]] = None, 
                **kwargs) -> Tuple[str, float]:
        """
        Generate a response using the transformer model
        
        Args:
            input_text: The text to respond to
            context: Optional conversation history
            kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated text, confidence score)
        """
        if not self.initialized:
            self.initialize()
        
        # Format input with context if provided
        formatted_input = input_text
        if context:
            conversation = ""
            for turn in context[-5:]:  # Use last 5 turns 
                if "user" in turn:
                    conversation += f"User: {turn['user']}\n"
                if "assistant" in turn:
                    conversation += f"GAKR: {turn['assistant']}\n"
            formatted_input = f"{conversation}User: {input_text}\nGAKR:"
        else:
            formatted_input = f"User: {input_text}\nGAKR:"
        
        # Set generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 150)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(formatted_input, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "GAKR:" in generated_text:
                response = generated_text.split("GAKR:")[-1].strip()
            else:
                response = generated_text.replace(formatted_input, "").strip()
            
            # Confidence score calculation (placeholder)
            confidence = 0.85
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error generating with transformer: {e}")
            return "I'm sorry, I couldn't process your request properly. Please try again.", 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        if not self.initialized:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "initialized": False,
                "parameters": 0
            }
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": True,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "layers": self.model.config.n_layer if hasattr(self.model.config, "n_layer") else "unknown"
        }
