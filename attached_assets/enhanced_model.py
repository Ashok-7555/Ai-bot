"""
GAKR Chatbot - Enhanced Model Interface
This module provides an interface to the enhanced model for the GAKR chatbot.
"""

import os
import json
import pickle
import random
import logging
import re
import math
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnhancedModel:
    """
    Interface to the enhanced model for the GAKR chatbot.
    """
    
    def __init__(self, model_path: str = "./models/trained/simple_model.pkl"):
        """
        Initialize the enhanced model.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.word_embeddings = {}
        self.patterns = []
        self.examples = []
        self.initialized = False
        
        # Try to load the model
        self._load_model()
        
    def _load_model(self) -> None:
        """
        Load the trained model from disk.
        """
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                
                logger.info(f"Loaded enhanced model from {self.model_path}")
                
                # Extract model components
                self.word_embeddings = self.model.get("word_embeddings", {})
                self.patterns = self.model.get("patterns", [])
                self.examples = self.model.get("examples", [])
                
                metadata = self.model.get("metadata", {})
                logger.info(f"Model metadata: {metadata}")
                
                self.initialized = True
            else:
                logger.warning(f"Enhanced model not found at {self.model_path}")
                self._load_fallback()
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            self._load_fallback()
    
    def _load_fallback(self) -> None:
        """
        Load fallback model data from training_data.json.
        """
        try:
            if os.path.exists("training_data.json"):
                with open("training_data.json", "r") as f:
                    data = json.load(f)
                
                logger.info("Loaded fallback data from training_data.json")
                
                # Convert to simple model format
                self.examples = [{"input": item["input"], "output": item["output"]} for item in data]
                self.patterns = []
                
                self.initialized = True
            else:
                logger.error("No fallback data found")
                self.initialized = False
        except Exception as e:
            logger.error(f"Error loading fallback data: {e}")
            self.initialized = False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Jaccard similarity
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _find_most_similar_example(self, input_text: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the most similar example to the input text.
        
        Args:
            input_text: Input text
            
        Returns:
            Tuple of (most similar example, similarity score)
        """
        best_example = None
        best_score = -1
        
        for example in self.examples:
            score = self._calculate_similarity(input_text, example["input"])
            if score > best_score:
                best_score = score
                best_example = example
        
        return best_example, best_score
    
    def _find_matching_pattern(self, input_text: str) -> Optional[Dict]:
        """
        Find a pattern that matches the input text.
        
        Args:
            input_text: Input text
            
        Returns:
            Matching pattern or None if no match found
        """
        for pattern in self.patterns:
            common_words = pattern.get("common_words", [])
            if all(word in input_text.lower() for word in common_words):
                return pattern
        
        return None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def _expand_template(self, template: str, input_text: str) -> str:
        """
        Expand a template with input-specific variations.
        
        Args:
            template: Template string
            input_text: Input text
            
        Returns:
            Expanded template
        """
        # Extract key entities from input
        input_words = re.findall(r'\b\w+\b', input_text.lower())
        
        # Extract important words (non-stopwords)
        key_words = [w for w in input_words if len(w) > 3 and w not in [
            "what", "when", "where", "which", "that", "this", "then", "than", 
            "will", "would", "could", "about", "there", "their", "they", "have"
        ]]
        
        result = template
        
        # Replace {{TOPIC}} with an important word from input if present
        if "{{TOPIC}}" in result and key_words:
            result = result.replace("{{TOPIC}}", random.choice(key_words))
        
        # Replace {{USER}} with a detected name
        name_match = re.search(r"my name is (\w+)", input_text.lower())
        if name_match and "{{USER}}" in result:
            result = result.replace("{{USER}}", name_match.group(1).capitalize())
        
        # Replace other placeholders
        import datetime
        now = datetime.datetime.now()
        
        placeholders = {
            "{{TIME}}": now.strftime("%H:%M:%S"),
            "{{DATE}}": now.strftime("%B %d, %Y"),
            "{{GREETING}}": random.choice(["Hello", "Hi", "Greetings", "Hey there"]),
            "{{USER}}": "you"  # Default if no name found
        }
        
        for placeholder, replacement in placeholders.items():
            result = result.replace(placeholder, replacement)
        
        return result
    
    def generate_response(self, input_text: str, conversation_history: Optional[List] = None) -> str:
        """
        Generate a response for the input text.
        
        Args:
            input_text: Input text
            conversation_history: Optional conversation history
            
        Returns:
            Generated response
        """
        if not self.initialized:
            return "I'm still learning how to generate better responses. Please try again later."
        
        try:
            # Preprocess input
            processed_input = self._preprocess_text(input_text)
            
            # Check for pattern match
            pattern = self._find_matching_pattern(processed_input)
            if pattern and pattern.get("response_templates"):
                response_template = random.choice(pattern["response_templates"])
                return self._expand_template(response_template, processed_input)
            
            # Find similar example
            example, score = self._find_most_similar_example(processed_input)
            if score > 0.4 and example:  # Good match
                return self._expand_template(example["output"], processed_input)
            
            # Check specific topic keywords
            topic_responses = {
                "hello": ["Hi there! How can I help you today?", "Hello! What can I assist you with?"],
                "help": ["I'm here to help! What do you need assistance with?", "I'd be happy to help. What would you like to know?"],
                "thanks": ["You're welcome!", "Happy to help!", "Anytime!"],
                "bye": ["Goodbye! Have a great day!", "See you later!", "Take care!"],
                "name": ["I am GAKR, an AI chatbot built to process and analyze text without external API dependencies.", 
                         "My name is GAKR. I'm a chatbot designed to help answer questions and have conversations."]
            }
            
            for topic, responses in topic_responses.items():
                if topic in processed_input:
                    return random.choice(responses)
            
            # Fallback to generic response
            generic_responses = [
                "That's an interesting topic. Can you tell me more?",
                "I understand what you're asking about. Could you provide more details?",
                "I'm processing that information. What specifically would you like to know?",
                "That's a good question. I'm thinking about the best way to answer.",
                "I'm learning more about topics like that. Can you be more specific about what you'd like to know?"
            ]
            
            return random.choice(generic_responses)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."


# Singleton instance
_enhanced_model = None

def get_enhanced_model():
    """
    Get the singleton enhanced model instance.
    
    Returns:
        EnhancedModel instance
    """
    global _enhanced_model
    
    if _enhanced_model is None:
        _enhanced_model = EnhancedModel()
    
    return _enhanced_model


def generate_enhanced_response(input_text: str, conversation_history: Optional[List] = None) -> str:
    """
    Generate a response using the enhanced model.
    
    Args:
        input_text: Input text
        conversation_history: Optional conversation history
        
    Returns:
        Generated response
    """
    model = get_enhanced_model()
    return model.generate_response(input_text, conversation_history)


if __name__ == "__main__":
    # Simple test
    test_inputs = [
        "What is your name?",
        "Tell me about artificial intelligence",
        "What is machine learning?",
        "How does a computer work?",
        "What is the difference between HTML and CSS?"
    ]
    
    model = EnhancedModel()
    
    if model.initialized:
        print("Enhanced model initialized successfully")
        for test_input in test_inputs:
            response = model.generate_response(test_input)
            print(f"\nInput: {test_input}")
            print(f"Response: {response}")
    else:
        print("Enhanced model initialization failed")