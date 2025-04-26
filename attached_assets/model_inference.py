"""
GAKR AI - Model Inference Module
This module handles the model inference process for the GAKR AI chatbot.
"""

import os
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Union

from core.utils.text_processing import process_input, process_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    """
    Handles inference for language models, coordinating the input processing,
    model prediction, and output processing steps.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = "enhanced"):
        """
        Initialize the model inference engine.
        
        Args:
            model_path: Path to the model files
            model_name: Name of the model to use
        """
        self.model_path = model_path or "./models/trained/simple_model.pkl"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.knowledge_base = self._load_knowledge_base()
        self.initialized = False
        
        # Try to initialize the model
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize model: {e}")
            logger.info("Will use fallback mechanisms for responses")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base from file.
        
        Returns:
            Dictionary containing knowledge base data
        """
        knowledge_base = {}
        
        try:
            # Try to load AI technical data 
            with open('ai_technical_data_fixed.json', 'r') as f:
                data = json.load(f)
                # Ensure we have a dictionary, not a list
                if isinstance(data, dict):
                    knowledge_base.update(data)
                elif isinstance(data, list) and len(data) > 0:
                    # Convert list to dictionary if possible
                    for item in data:
                        if isinstance(item, dict) and 'key' in item and 'value' in item:
                            knowledge_base[item['key']] = item['value']
        except Exception as e:
            logger.warning(f"Failed to load AI technical data: {e}")
            
        if not knowledge_base:
            # Try to load training data as fallback
            try:
                with open('training_data.json', 'r') as f:
                    data = json.load(f)
                    # Handle the case of a list properly
                    if isinstance(data, dict):
                        knowledge_base.update(data)
                    elif isinstance(data, list) and len(data) > 0:
                        # Basic conversion strategy for list of examples
                        knowledge_base["examples"] = data
                        # Extract topics from examples if possible
                        topics = {}
                        for item in data:
                            if isinstance(item, dict) and 'topics' in item:
                                for topic in item.get('topics', []):
                                    topics[topic] = topics.get(topic, 0) + 1
                        knowledge_base["topics"] = topics
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")
        
        # Add some basic knowledge if nothing was loaded
        if not knowledge_base:
            knowledge_base = {
                "ai": "Artificial Intelligence (AI) refers to systems designed to perform tasks that typically require human intelligence.",
                "gakr": "GAKR is an AI chatbot that can process and analyze text using various models.",
                "chatbot": "A chatbot is a computer program designed to simulate conversation with human users."
            }
            
        return knowledge_base
    
    def _initialize_model(self) -> None:
        """
        Initialize the model based on model_name.
        
        Different model types will be initialized differently:
        - "simple": Use regex pattern matching and basic templating
        - "enhanced": Use more advanced pattern matching with similarity scoring
        - "neural": Try to load a neural model for inference
        """
        if self.model_name == "simple":
            # Simple models don't need special initialization
            self.initialized = True
            logger.info("Simple pattern matching model initialized")
            
        elif self.model_name == "enhanced":
            # Try to import and initialize the enhanced model
            try:
                from enhanced_model import EnhancedModel
                self.model = EnhancedModel(self.model_path)
                self.initialized = True
                logger.info("Enhanced model initialized successfully")
            except ImportError:
                logger.warning("Enhanced model module not found, falling back to simple model")
                self.model_name = "simple"
                self.initialized = True
        
        else:
            logger.warning(f"Unknown model type {self.model_name}, using simple model instead")
            self.model_name = "simple"
            self.initialized = True
    
    def _get_simple_response(self, input_data: Dict[str, Any]) -> str:
        """
        Generate a response using simple pattern matching.
        
        Args:
            input_data: Processed input data
            
        Returns:
            Generated response
        """
        text = input_data["text"]
        
        # Check knowledge base for matching information
        for term, info in self.knowledge_base.items():
            if isinstance(info, str) and term.lower() in text.lower():
                return info
        
        # Check for common question patterns
        if any(q in text.lower() for q in ["what is", "how does", "how do", "explain", "describe"]):
            topic = text.lower().replace("what is", "").replace("how does", "").replace("how do", "").replace("explain", "").replace("describe", "").strip()
            
            if "gakr" in topic or "chatbot" in topic or "you" in topic:
                return ("I am GAKR AI, an advanced chatbot powered by a combination of "
                        "pattern matching, knowledge base retrieval, and neural models. "
                        "I can process text, answer questions, and engage in conversations "
                        "across various topics including AI, programming, and data science.")
            
            if "ai" in topic or "artificial intelligence" in topic:
                return ("Artificial Intelligence (AI) refers to computer systems designed to perform tasks "
                        "that typically require human intelligence. These include learning, reasoning, "
                        "problem-solving, perception, and language understanding. AI systems can be rule-based "
                        "or use machine learning techniques to improve over time through experience.")
        
        # Default responses if no match is found
        default_responses = [
            "I understand you're asking about that topic, but I don't have enough specific information in my knowledge base.",
            "That's an interesting question. While I don't have detailed information on that, I can help with AI-related topics.",
            "I'm still learning about that subject. Could you ask me something else related to AI or programming?",
            "I don't have enough context to properly answer that question. Could you provide more details?"
        ]
        
        return random.choice(default_responses)
    
    def _get_enhanced_response(self, input_data: Dict[str, Any]) -> str:
        """
        Generate a response using the enhanced model.
        
        Args:
            input_data: Processed input data
            
        Returns:
            Generated response
        """
        if not self.model:
            logger.warning("Enhanced model not properly initialized, falling back to simple model")
            return self._get_simple_response(input_data)
        
        try:
            # Convert context to format expected by enhanced model
            context_history = []
            if input_data["context"]:
                for line in input_data["context"].split('\n'):
                    if line.startswith("User:"):
                        context_history.append({"type": "user", "message": line[5:].strip()})
                    elif line.startswith("Assistant:"):
                        context_history.append({"type": "bot", "message": line[10:].strip()})
            
            # Generate response using enhanced model
            response = self.model.generate_response(input_data["text"], context_history)
            return response
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
            return self._get_simple_response(input_data)
    
    def generate_response(self, user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None, 
                          system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for user input.
        
        Args:
            user_input: The raw user input text
            conversation_history: Previous conversation exchanges
            system_prompt: Optional system instructions
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Process the input
            processed_input = process_input(user_input, conversation_history, system_prompt)
            
            # Generate raw response
            if self.model_name == "simple":
                raw_response = self._get_simple_response(processed_input)
            elif self.model_name == "enhanced":
                raw_response = self._get_enhanced_response(processed_input)
            else:
                raw_response = self._get_simple_response(processed_input)
            
            # Post-process the response
            processed_response = process_output(raw_response)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare full response with metadata
            full_response = {
                "response": processed_response,
                "model": self.model_name,
                "generation_time": generation_time,
                "input_tokens": len(processed_input["tokens"]),
                # Include analysis data
                "analysis_type": "text_generation",
                "sentiment": self._analyze_sentiment(user_input),
                "confidence": self._calculate_confidence(user_input, processed_response)
            }
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while generating a response.",
                "model": self.model_name,
                "error": str(e)
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform basic sentiment analysis on input text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Simple sentiment analysis with keyword matching
        positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "wonderful", "awesome", "fantastic", 
                          "nice", "pleased", "glad", "joy", "exciting", "brilliant", "terrific", "perfect", "impressive", "beautiful"]
        negative_words = ["bad", "terrible", "awful", "hate", "poor", "sad", "horrible", "dislike", "disappointed", 
                          "upset", "angry", "frustrating", "annoying", "useless", "stupid", "boring", "worse", "worst", "not satisfied"]
        
        text = text.lower()
        words = text.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        sentiment = "neutral"
        score = 0.5
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.5 + (0.1 * positive_count)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.5 - (0.1 * negative_count)
        
        # Cap score between 0 and 1
        score = max(0, min(1, score))
        
        return {
            "sentiment": sentiment,
            "score": score
        }
    
    def _calculate_confidence(self, input_text: str, response: str) -> float:
        """
        Calculate a confidence score for the generated response.
        
        Args:
            input_text: User input
            response: Generated response
            
        Returns:
            Confidence score between 0 and 1
        """
        # This is a simplified confidence calculation
        # In a real implementation, this would use model-specific confidence scores
        
        # Higher confidence for longer, more detailed responses
        length_factor = min(1.0, len(response) / 100)
        
        # Higher confidence if response contains keywords from input
        input_words = set(input_text.lower().split())
        response_words = set(response.lower().split())
        
        overlap = input_words.intersection(response_words)
        overlap_factor = min(1.0, len(overlap) / max(1, len(input_words)))
        
        # Lower confidence for generic responses
        generic_phrases = ["I don't know", "I don't have", "I can't", "I'm sorry", "not enough information"]
        generic_factor = 1.0
        for phrase in generic_phrases:
            if phrase.lower() in response.lower():
                generic_factor *= 0.7
        
        # Calculate final confidence
        confidence = (0.4 * length_factor + 0.4 * overlap_factor + 0.2) * generic_factor
        
        return min(1.0, max(0.1, confidence))

# Create a singleton instance
model_inference = ModelInference()

def generate_response(user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None,
                     system_prompt: Optional[str] = None, model_name: str = "enhanced") -> Dict[str, Any]:
    """
    Global function to generate a response for user input.
    
    Args:
        user_input: User's message
        conversation_history: Previous conversation exchanges
        system_prompt: Optional system instructions
        model_name: Model to use for generation
        
    Returns:
        Dictionary with response and metadata
    """
    # Create a model inference instance with the specified model
    inference = ModelInference(model_name=model_name)
    
    # Generate the response
    return inference.generate_response(user_input, conversation_history, system_prompt)