"""
GAKR Perplexity Integration Module
This module provides functions to integrate with the Perplexity API for enhanced responses.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PerplexityClient:
    """
    Client for interacting with the Perplexity API.
    """
    
    def __init__(self):
        """Initialize the Perplexity client."""
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai"
        self.is_available = self.api_key is not None
    
    def check_availability(self) -> bool:
        """
        Check if the Perplexity API is available.
        
        Returns:
            True if the API is available, False otherwise
        """
        if not self.api_key:
            logger.warning("Perplexity API key not found in environment")
            return False
        
        return True
    
    def complete(self, 
                messages: List[Dict[str, str]], 
                model: str = "llama-3.1-sonar-small-128k-online",
                temperature: float = 0.2,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Call the Perplexity chat completions API.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            model: Model to use for completion
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            API response
        """
        if not self.is_available:
            logger.error("Perplexity API not available")
            return {"error": "Perplexity API not available"}
        
        # Check if messages starts with a system message
        if not messages or messages[0].get("role") != "system":
            # Add a default system message
            messages.insert(0, {
                "role": "system",
                "content": "You are GAKR, a helpful, knowledgeable AI assistant that provides accurate and informative responses."
            })
        
        # Ensure messages alternate correctly and end with user
        if messages[-1].get("role") != "user":
            logger.warning("Messages must end with a user message. Ignoring last assistant message.")
            messages = [msg for idx, msg in enumerate(messages) if 
                      idx == 0 or  # Keep system message
                      (idx > 0 and msg.get("role") != messages[idx-1].get("role"))]  # Ensure alternating
            
            # If still doesn't end with user, something's wrong
            if messages[-1].get("role") != "user":
                logger.error("Failed to format messages correctly")
                return {"error": "Invalid message format"}
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "search_recency_filter": "week", # Updated frequently
                "return_related_questions": False,
                "stream": False
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Perplexity API: {e}")
            return {"error": str(e)}
    
    def generate_response(self, user_input: str, conversation_history: Optional[List] = None) -> str:
        """
        Generate a response using the Perplexity API.
        
        Args:
            user_input: User input text
            conversation_history: Optional conversation history
            
        Returns:
            Generated response or error message
        """
        if not self.is_available:
            return "I don't have access to the Perplexity API right now. Let me answer based on my local knowledge."
        
        # Build messages from conversation history
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are GAKR, a helpful AI assistant. Provide accurate, concise, and informative responses. Focus on facts and avoid speculation."
        })
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role in ["user", "assistant", "system"]:
                    messages.append({
                        "role": role,
                        "content": content
                    })
        
        # Add the current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Call the API
        response = self.complete(messages)
        
        # Extract the response text
        if "error" in response:
            logger.error(f"Error from Perplexity API: {response['error']}")
            return f"I encountered an error while accessing external knowledge: {response['error']}"
        
        try:
            assistant_message = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check if we got a meaningful response
            if not assistant_message:
                logger.warning("Empty response from Perplexity API")
                return "I couldn't generate a response using external knowledge. Let me try answering with my local knowledge."
            
            # Potentially add citation info
            citations = response.get("citations", [])
            if citations:
                citation_text = "\n\nSources:"
                for i, citation in enumerate(citations[:3], 1):  # Limit to first 3 citations
                    citation_text += f"\n{i}. {citation}"
                
                assistant_message += citation_text
            
            return assistant_message
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Perplexity API response: {e}")
            return "I had trouble processing the information. Let me answer based on my local knowledge instead."


# Singleton instance
_perplexity_client = None

def get_perplexity_client():
    """
    Get the singleton Perplexity client instance.
    
    Returns:
        PerplexityClient instance
    """
    global _perplexity_client
    
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    
    return _perplexity_client

def check_perplexity_availability():
    """
    Check if the Perplexity API is available.
    
    Returns:
        True if the API is available, False otherwise
    """
    client = get_perplexity_client()
    return client.check_availability()

def generate_perplexity_response(user_input: str, conversation_history: Optional[List] = None) -> str:
    """
    Generate a response using the Perplexity API.
    
    Args:
        user_input: User input text
        conversation_history: Optional conversation history
        
    Returns:
        Generated response
    """
    client = get_perplexity_client()
    return client.generate_response(user_input, conversation_history)


if __name__ == "__main__":
    # Simple test
    if check_perplexity_availability():
        print("Perplexity API is available")
        response = generate_perplexity_response("What is artificial intelligence?")
        print(f"Response: {response}")
    else:
        print("Perplexity API is not available. Make sure PERPLEXITY_API_KEY is set in the environment.")