import re
import unicodedata
import logging
from typing import List, Dict, Any, Optional

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Handles text preprocessing and normalization for GAKR AI
    """
    
    def __init__(self):
        """Initialize the text processor"""
        self.stop_words = set(stopwords.words('english'))
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def clean_for_embedding(self, text: str) -> str:
        """
        Clean text specifically for embedding models
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text ready for embedding
        """
        # Normalize
        text = self.normalize(text)
        
        # Tokenize and remove stopwords
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Rejoin into a single string
        cleaned_text = ' '.join(filtered_tokens)
        
        return cleaned_text
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query for the chatbot
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary with processed information
        """
        normalized = self.normalize(query)
        tokens = self.tokenize(normalized)
        
        return {
            'original': query,
            'normalized': normalized,
            'tokens': tokens,
            'token_count': len(tokens)
        }
