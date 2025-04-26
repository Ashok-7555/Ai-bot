"""
GAKR Chatbot - Simple Neural Response Generator
A lightweight alternative to transformer models for generating varied responses.
"""

import json
import random
import logging
import os
import re
from collections import defaultdict
import math
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SimpleNeuralGenerator:
    """
    A simple neural-inspired text generator that creates varied responses
    based on training data patterns without requiring heavy ML libraries.
    """
    
    def __init__(self, training_data_path: str = "training_data.json"):
        """
        Initialize the neural generator.
        
        Args:
            training_data_path: Path to the JSON training data
        """
        self.training_data_path = training_data_path
        self.patterns = defaultdict(list)
        self.word_embeddings = {}
        self.topic_keywords = defaultdict(list)
        self.initialized = False
        self.vocabulary = set()
        self.response_templates = []
        
        # Initialize the model
        self._load_and_process_data()
    
    def _load_and_process_data(self) -> None:
        """
        Load and process the training data.
        """
        try:
            if not os.path.exists(self.training_data_path):
                logger.warning(f"Training data file {self.training_data_path} not found.")
                self.initialized = False
                return
            
            logger.info(f"Loading training data from {self.training_data_path}")
            with open(self.training_data_path, "r") as f:
                data = json.load(f)
            
            logger.info(f"Processing {len(data)} training examples")
            
            # Build vocabulary and extract patterns
            all_words = set()
            for item in data:
                input_text = item["input"].lower()
                output_text = item["output"]
                
                # Extract words from input
                input_words = re.findall(r'\b\w+\b', input_text)
                all_words.update(input_words)
                
                # Store as response template
                self.response_templates.append({
                    "input_pattern": input_text,
                    "output_template": output_text,
                    "input_words": input_words
                })
                
                # Extract topic keywords
                for word in input_words:
                    if len(word) > 3 and word not in ["what", "when", "where", "which", "that", "this", "then", "than", 
                                                       "will", "would", "could", "about", "there", "their", "they", "have"]:
                        self.topic_keywords[word].append(item["output"])
            
            self.vocabulary = all_words
            logger.info(f"Built vocabulary with {len(self.vocabulary)} words")
            
            # Create simple "word embeddings" (just word co-occurrence)
            for item in data:
                input_words = re.findall(r'\b\w+\b', item["input"].lower())
                for i, word in enumerate(input_words):
                    context_words = input_words[max(0, i-2):i] + input_words[i+1:min(len(input_words), i+3)]
                    if word not in self.word_embeddings:
                        self.word_embeddings[word] = defaultdict(int)
                    for context_word in context_words:
                        self.word_embeddings[word][context_word] += 1
            
            logger.info("Neural generator initialized successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing neural generator: {e}")
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
    
    def _find_most_similar_template(self, input_text: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the most similar template to the input text.
        
        Args:
            input_text: The input text
            
        Returns:
            Tuple of (most similar template, similarity score)
        """
        best_template = None
        best_score = -1
        
        for template in self.response_templates:
            score = self._calculate_similarity(input_text, template["input_pattern"])
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template, best_score
    
    def _expand_template(self, template: str, input_text: str) -> str:
        """
        Expand a template with input-specific variations.
        
        Args:
            template: The template string
            input_text: The input text
            
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
        
        # Replace other placeholders
        placeholders = {
            "{{TIME}}": "the current time",
            "{{DATE}}": "today's date",
            "{{GREETING}}": random.choice(["Hello", "Hi", "Greetings", "Hey there"]),
            "{{USER}}": "you"
        }
        
        for placeholder, replacement in placeholders.items():
            result = result.replace(placeholder, replacement)
        
        return result
    
    def generate_response(self, input_text: str) -> str:
        """
        Generate a response for the given input text.
        
        Args:
            input_text: The input text
            
        Returns:
            Generated response
        """
        if not self.initialized:
            return "I'm still learning how to generate better responses. Please try again later."
        
        try:
            # Find similar template
            best_template, score = self._find_most_similar_template(input_text)
            
            if score > 0.3 and best_template is not None:  # Good match
                return self._expand_template(best_template["output_template"], input_text)
            
            # Try topic-based response
            input_words = re.findall(r'\b\w+\b', input_text.lower())
            relevant_responses = []
            
            for word in input_words:
                if word in self.topic_keywords and len(self.topic_keywords[word]) > 0:
                    relevant_responses.extend(self.topic_keywords[word])
            
            if relevant_responses:
                return random.choice(relevant_responses)
            
            # Fallback to generic response
            generic_responses = [
                "That's an interesting point. Can you tell me more?",
                "I understand what you're asking about. Let me think about that.",
                "I'm not entirely sure about that. Could you provide more details?",
                "That's a good question. I'm processing that information.",
                "I'm still learning about topics like this. What specifically would you like to know?"
            ]
            
            return random.choice(generic_responses)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."


# Singleton instance
_neural_generator = None

def get_neural_generator():
    """
    Get the singleton neural generator instance.
    
    Returns:
        SimpleNeuralGenerator instance
    """
    global _neural_generator
    
    if _neural_generator is None:
        _neural_generator = SimpleNeuralGenerator()
    
    return _neural_generator


def generate_neural_response(input_text: str) -> str:
    """
    Generate a response using the neural generator.
    
    Args:
        input_text: The input text
        
    Returns:
        Generated response text
    """
    generator = get_neural_generator()
    return generator.generate_response(input_text)


if __name__ == "__main__":
    # Simple test
    test_inputs = [
        "What is your name?",
        "Tell me about artificial intelligence",
        "What is machine learning?",
        "How does a computer work?",
        "What is the difference between HTML and CSS?"
    ]
    
    generator = SimpleNeuralGenerator()
    
    if generator.initialized:
        print("Neural generator initialized successfully")
        for test_input in test_inputs:
            response = generator.generate_response(test_input)
            print(f"\nInput: {test_input}")
            print(f"Response: {response}")
    else:
        print("Neural generator initialization failed")