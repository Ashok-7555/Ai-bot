"""
GAKR AI - Token Encoder Module
This module handles token encoding, embedding, and related operations.
"""

import os
import json
import logging
import random
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenEncoder:
    """
    Token encoding and embedding operations for language models.
    Provides functionality to convert text to token IDs and vice versa.
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize the token encoder.
        
        Args:
            vocab_path: Path to the vocabulary file
        """
        self.vocab_path = vocab_path
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4,
            "<user>": 5,
            "<assistant>": 6,
            "<system>": 7
        }
        
        # Load vocabulary or create a basic one
        self._load_or_create_vocab()
    
    def _load_or_create_vocab(self) -> None:
        """Load the vocabulary from file or create a basic one."""
        if self.vocab_path and os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, 'r') as f:
                    self.vocab = json.load(f)
                    
                # Create inverse vocab mapping
                self.inverse_vocab = {id: token for token, id in self.vocab.items()}
                logger.info(f"Loaded vocabulary with {len(self.vocab)} tokens")
                return
            except Exception as e:
                logger.warning(f"Failed to load vocabulary from {self.vocab_path}: {e}")
        
        # Create a basic vocabulary with special tokens and common English words
        self.vocab = self.special_tokens.copy()
        
        # Add some common English words to the basic vocabulary
        common_words = ["the", "a", "an", "is", "are", "was", "were", "and", "or", "but", 
                       "I", "you", "he", "she", "it", "we", "they", "my", "your", "this", 
                       "that", "these", "those", "what", "who", "how", "why", "when", "where",
                       "hello", "hi", "yes", "no", "thank", "please", "help", "can", "would", 
                       "will", "should", "could", "do", "does", "did", "has", "have", "had"]
        
        # Add common words to vocabulary
        token_id = len(self.vocab)
        for word in common_words:
            if word not in self.vocab:
                self.vocab[word] = token_id
                token_id += 1
        
        # Create inverse vocab mapping
        self.inverse_vocab = {id: token for token, id in self.vocab.items()}
        logger.info(f"Created basic vocabulary with {len(self.vocab)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add start and end tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Tokenize by words (very simple implementation)
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Convert tokens to IDs
        ids = []
        if add_special_tokens:
            ids.append(self.vocab["<s>"])
        
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<unk>"])
        
        if add_special_tokens:
            ids.append(self.vocab["</s>"])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not ids:
            return ""
        
        # Decode IDs to tokens
        tokens = []
        for id in ids:
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append("<unk>")
        
        # Convert tokens to text (simple space joining)
        # In a real implementation, this would handle subwords and proper spacing
        text = " ".join(tokens)
        
        # Clean up spaces around punctuation
        text = re.sub(r'\s+([,.!?:;])', r'\1', text)
        
        return text
    
    def save_vocab(self, path: str) -> bool:
        """
        Save the vocabulary to file.
        
        Args:
            path: Path to save the vocabulary
            
        Returns:
            Success flag
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.vocab, f, indent=2)
            logger.info(f"Saved vocabulary with {len(self.vocab)} tokens to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vocabulary to {path}: {e}")
            return False
    
    def add_tokens(self, tokens: List[str]) -> int:
        """
        Add new tokens to the vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            Number of tokens added
        """
        added = 0
        token_id = len(self.vocab)
        
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = token_id
                self.inverse_vocab[token_id] = token
                token_id += 1
                added += 1
        
        return added

class EmbeddingManager:
    """
    Manages token embeddings and performs vector operations.
    """
    
    def __init__(self, embedding_size: int = 50, encoder: Optional[TokenEncoder] = None):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_size: Dimensionality of embeddings
            encoder: TokenEncoder instance
        """
        self.embedding_size = embedding_size
        self.encoder = encoder or TokenEncoder()
        self.embeddings = {}
        
        # Initialize embeddings for special tokens
        for token, id in self.encoder.special_tokens.items():
            self.embeddings[id] = self._create_random_embedding()
    
    def _create_random_embedding(self) -> List[float]:
        """
        Create a random embedding vector.
        
        Returns:
            Random embedding vector
        """
        return [random.uniform(-0.1, 0.1) for _ in range(self.embedding_size)]
    
    def get_embedding(self, token_id: int) -> List[float]:
        """
        Get the embedding for a token ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Embedding vector
        """
        if token_id in self.embeddings:
            return self.embeddings[token_id]
        
        # Create and store a new embedding if not found
        self.embeddings[token_id] = self._create_random_embedding()
        return self.embeddings[token_id]
    
    def get_embeddings_for_text(self, text: str) -> List[List[float]]:
        """
        Get embeddings for all tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            List of embedding vectors
        """
        token_ids = self.encoder.encode(text)
        return [self.get_embedding(id) for id in token_ids]
    
    def save_embeddings(self, path: str) -> bool:
        """
        Save embeddings to file.
        
        Args:
            path: Path to save embeddings
            
        Returns:
            Success flag
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Saved {len(self.embeddings)} embeddings to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embeddings to {path}: {e}")
            return False
    
    def load_embeddings(self, path: str) -> bool:
        """
        Load embeddings from file.
        
        Args:
            path: Path to load embeddings from
            
        Returns:
            Success flag
        """
        try:
            with open(path, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self.embeddings)} embeddings from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embeddings from {path}: {e}")
            return False

# Create singleton instances for convenience
token_encoder = TokenEncoder()
embedding_manager = EmbeddingManager(encoder=token_encoder)

def encode_text(text: str, add_special_tokens: bool = True) -> List[int]:
    """
    Encode text into token IDs.
    
    Args:
        text: Text to encode
        add_special_tokens: Whether to add special tokens
        
    Returns:
        List of token IDs
    """
    return token_encoder.encode(text, add_special_tokens)

def decode_ids(ids: List[int], skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs into text.
    
    Args:
        ids: List of token IDs
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded text
    """
    return token_encoder.decode(ids, skip_special_tokens)

def get_embeddings(text: str) -> List[List[float]]:
    """
    Get embeddings for text.
    
    Args:
        text: Input text
        
    Returns:
        List of embedding vectors
    """
    return embedding_manager.get_embeddings_for_text(text)