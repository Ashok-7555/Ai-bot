"""
GAKR AI - Text Processing Module
This module handles the text processing pipeline for the GAKR AI chatbot.
"""

import re
import string
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processing pipeline for chat input and response generation.
    Handles various preprocessing steps for text before passing to a language model
    and post-processing steps for generated responses.
    """
    
    def __init__(self):
        """Initialize the text processor with default configuration."""
        self.stop_words = set([
            "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "in",
            "of", "is", "am", "are", "was", "were", "be", "being", "been", "have", "has", "had",
            "do", "does", "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could"
        ])
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.html_tag_pattern = re.compile(r'<.*?>')
        self.extra_whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'([%s])' % re.escape(string.punctuation))
    
    def preprocess_text(self, text: str, clean_level: str = "medium") -> str:
        """
        Preprocess input text with configurable cleaning levels.
        
        Args:
            text: Input text from user
            clean_level: Cleaning intensity level - 'light', 'medium', or 'aggressive'
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Always perform basic cleaning
        cleaned_text = text.strip()
        cleaned_text = self.html_tag_pattern.sub(' ', cleaned_text)  # Remove HTML tags
        
        if clean_level in ["medium", "aggressive"]:
            # Medium cleaning
            cleaned_text = self.url_pattern.sub('[URL]', cleaned_text)  # Replace URLs
            cleaned_text = self.email_pattern.sub('[EMAIL]', cleaned_text)  # Replace emails
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespace
        
        if clean_level == "aggressive":
            # Aggressive cleaning - lowercase and remove punctuation
            cleaned_text = cleaned_text.lower()
            
            # Add spaces around punctuation for better tokenization
            cleaned_text = self.punctuation_pattern.sub(r' \1 ', cleaned_text)
            
            # Remove stop words (for some applications)
            # words = cleaned_text.split()
            # words = [word for word in words if word.lower() not in self.stop_words]
            # cleaned_text = ' '.join(words)
            
            # Final whitespace cleanup
            cleaned_text = self.extra_whitespace_pattern.sub(' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (words and punctuation).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Simple whitespace tokenization
        tokens = self.punctuation_pattern.sub(r' \1 ', text)
        tokens = self.extra_whitespace_pattern.sub(' ', tokens)
        return tokens.split()
    
    def format_conversation_history(self, history: List[Dict[str, Any]],
                                   max_context_length: int = 5) -> str:
        """
        Format conversation history into a single string for context.
        
        Args:
            history: List of conversation history entries
            max_context_length: Maximum number of previous exchanges to include
            
        Returns:
            Formatted conversation history
        """
        if not history:
            return ""
        
        # Get the most recent conversation exchanges
        recent_history = history[-max_context_length:] if len(history) > max_context_length else history
        
        # Format history into a string with user/assistant roles
        formatted_history = []
        for entry in recent_history:
            role = "User:" if entry.get("type") == "user" else "Assistant:"
            message = entry.get("message", "")
            formatted_history.append(f"{role} {message}")
        
        return "\n".join(formatted_history)
    
    def prepare_model_input(self, text: str, conversation_history: Optional[List[Dict[str, Any]]] = None,
                           system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare the final input for the language model.
        
        Args:
            text: Current user input
            conversation_history: Previous conversation history
            system_prompt: Optional system instructions
            
        Returns:
            Dictionary with prepared model inputs
        """
        preprocessed_text = self.preprocess_text(text)
        tokens = self.tokenize(preprocessed_text)
        
        context = ""
        if conversation_history:
            context = self.format_conversation_history(conversation_history)
        
        prepared_input = {
            "text": preprocessed_text,
            "tokens": tokens,
            "context": context
        }
        
        if system_prompt:
            prepared_input["system_prompt"] = system_prompt
        
        return prepared_input
    
    def post_process_response(self, response: str) -> str:
        """
        Post-process the model's response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Processed response
        """
        if not response:
            return ""
        
        # Clean up whitespace
        processed = response.strip()
        processed = self.extra_whitespace_pattern.sub(' ', processed)
        
        # Fix common formatting issues
        processed = re.sub(r'\s+([.,!?;:])', r'\1', processed)  # Remove spaces before punctuation
        
        # Fix capitalization if needed
        if len(processed) > 0 and processed[0].islower():
            processed = processed[0].upper() + processed[1:]
        
        # Make sure response ends with proper punctuation
        if len(processed) > 0 and processed[-1] not in ".!?":
            processed += "."
            
        return processed

# Create a singleton instance
processor = TextProcessor()

def process_input(user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None,
                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Process user input through the complete preprocessing pipeline.
    
    Args:
        user_input: User's message
        conversation_history: Previous conversation history
        system_prompt: Optional system instructions
        
    Returns:
        Processed input ready for the model
    """
    return processor.prepare_model_input(user_input, conversation_history, system_prompt)

def process_output(model_response: Union[str, Dict[str, Any]]) -> str:
    """
    Process model output through the post-processing pipeline.
    
    Args:
        model_response: Raw response from the model, either as string or dictionary
        
    Returns:
        Processed response ready for display
    """
    if isinstance(model_response, dict) and "response" in model_response:
        return processor.post_process_response(str(model_response.get("response", "")))
    return processor.post_process_response(str(model_response))