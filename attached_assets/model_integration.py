"""
GAKR Chatbot - Model Integration Module
This module integrates the trained transformer models with the GAKR chatbot.
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TransformerResponseGenerator:
    """
    A class that uses a fine-tuned transformer model to generate responses
    for the GAKR chatbot.
    """
    
    def __init__(self, model_path="./models/trained"):
        """
        Initialize the transformer response generator.
        
        Args:
            model_path: Path to the trained model and tokenizer
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.max_length = 150
        self.initialized = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the transformer model and tokenizer.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model path {self.model_path} does not exist. Transformer integration will not be available.")
                return False
            
            logger.info(f"Loading transformer model from {self.model_path}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            logger.info(f"Successfully loaded transformer model to {device}")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return False
    
    def generate_response(self, user_input, conversation_history=None):
        """
        Generate a response using the transformer model.
        
        Args:
            user_input: The user's input text
            conversation_history: A list of previous conversation turns (optional)
            
        Returns:
            Generated response text or None if generation fails
        """
        if not self.initialized:
            logger.warning("Transformer model not initialized. Cannot generate response.")
            return None
        
        try:
            # Format input based on whether we have conversation history
            if conversation_history:
                # Format as a multi-turn conversation
                prompt = ""
                for i, (user_msg, bot_msg) in enumerate(conversation_history[-3:]):  # Use last 3 turns
                    prompt += f"User: {user_msg}\nGAKR: {bot_msg}\n"
                prompt += f"User: {user_input}\nGAKR:"
            else:
                # Single turn
                prompt = f"User: {user_input}\nGAKR:"
            
            # Encode the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to same device as model
            input_ids = input_ids.to(self.model.device)
            
            # Generate response
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + self.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the bot's response after "GAKR:"
            response_parts = generated_text.split("GAKR:")
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
                
                # Remove any trailing "User:" and beyond
                if "User:" in response:
                    response = response.split("User:")[0].strip()
                
                return response
            else:
                # Fallback if format is unexpected
                return generated_text
                
        except Exception as e:
            logger.error(f"Error generating transformer response: {e}")
            return None


# Singleton instance
transformer_generator = None

def get_transformer_generator(model_path="./models/trained"):
    """
    Get or create the transformer response generator.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        TransformerResponseGenerator instance
    """
    global transformer_generator
    
    if transformer_generator is None:
        transformer_generator = TransformerResponseGenerator(model_path)
    
    return transformer_generator


def generate_transformer_response(user_input, conversation_history=None):
    """
    Generate a response using the transformer model.
    
    Args:
        user_input: The user's input text
        conversation_history: A list of previous conversation turns (optional)
        
    Returns:
        Generated response text or None if generation fails
    """
    generator = get_transformer_generator()
    return generator.generate_response(user_input, conversation_history)


if __name__ == "__main__":
    # Simple test
    generator = get_transformer_generator()
    
    if generator.initialized:
        test_input = "What is your name?"
        response = generator.generate_response(test_input)
        print(f"Input: {test_input}")
        print(f"Response: {response}")
    else:
        print("Transformer model not initialized. Please train the model first.")
        print("Run model_training.py to train the model.")