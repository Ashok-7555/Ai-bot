"""
Text preprocessing utilities for the GAKR chatbot.
"""

import re
import string
from typing import List, Optional

def remove_special_characters(text: str) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Keep alphanumeric, space, and basic punctuation
    return re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length to truncate to
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    sentences = text[:max_length].split('.')
    if len(sentences) > 1:
        return '.'.join(sentences[:-1]) + '.'
    
    # If no sentence boundary found, truncate at word boundary
    words = text[:max_length].split()
    if len(words) > 1:
        return ' '.join(words[:-1])
    
    # If all else fails, just truncate
    return text[:max_length]

def preprocess_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Apply preprocessing steps to input text.
    
    Args:
        text: Input text
        max_length: Optional maximum length for truncation
        
    Returns:
        Preprocessed text
    """
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Apply preprocessing steps
    text = text.strip()
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    
    # Truncate if max_length provided
    if max_length:
        text = truncate_text(text, max_length)
    
    return text
