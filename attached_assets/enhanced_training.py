"""
GAKR Chatbot - Enhanced Training Module
This module provides a lighter-weight training approach using available packages on Replit
"""

import json
import random
import logging
import os
import re
import pickle
from collections import defaultdict
import math
import time
import csv
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """
    Processes and prepares data for training the GAKR chatbot.
    Uses a lightweight approach that doesn't require heavy ML libraries.
    """
    
    def __init__(self, output_path: str = "processed_data.json"):
        """
        Initialize the data processor.
        
        Args:
            output_path: Path to save processed data
        """
        self.output_path = output_path
        self.training_examples = []
        self.vocabulary = set()
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_embeddings = {}
        
    def add_example(self, input_text: str, output_text: str, meta: Dict = None) -> None:
        """
        Add a single training example.
        
        Args:
            input_text: Input text (user message)
            output_text: Output text (bot response)
            meta: Optional metadata for the example
        """
        if not meta:
            meta = {"source": "custom", "category": "general"}
            
        example = {
            "input": input_text,
            "output": output_text,
            "meta": meta
        }
        
        self.training_examples.append(example)
        
        # Update vocabulary with new words
        for text in [input_text, output_text]:
            words = self._tokenize(text)
            self.vocabulary.update(words)
            
        logger.info(f"Added training example, total examples: {len(self.training_examples)}")
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'"-]', ' ', text)
        # Split on whitespace
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
        
    def load_custom_data(self, path: str) -> None:
        """
        Load custom training data in GAKR format.
        
        Args:
            path: Path to the data file
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} custom training examples from {path}")
            
            for item in data:
                self.training_examples.append({
                    "input": item["input"],
                    "output": item["output"],
                    "source": "custom"
                })
                
                # Update vocabulary
                words = re.findall(r'\b\w+\b', item["input"].lower())
                self.vocabulary.update(words)
        
        except Exception as e:
            logger.error(f"Error loading custom data from {path}: {e}")
    
    def load_conversation_csv(self, path: str, input_col: int = 0, output_col: int = 1, 
                             has_header: bool = True, max_examples: int = 10000) -> None:
        """
        Load conversation data from CSV file.
        
        Args:
            path: Path to the CSV file
            input_col: Index of input column
            output_col: Index of output column
            has_header: Whether the CSV has a header row
            max_examples: Maximum number of examples to load
        """
        try:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                
                if has_header:
                    next(reader)  # Skip header
                
                for row in reader:
                    if len(row) > max(input_col, output_col):
                        input_text = row[input_col].strip()
                        output_text = row[output_col].strip()
                        
                        if input_text and output_text:
                            self.training_examples.append({
                                "input": input_text,
                                "output": output_text,
                                "source": os.path.basename(path)
                            })
                            
                            # Update vocabulary
                            words = re.findall(r'\b\w+\b', input_text.lower())
                            self.vocabulary.update(words)
                            
                            count += 1
                            if count >= max_examples:
                                break
            
            logger.info(f"Loaded {count} conversation examples from {path}")
        
        except Exception as e:
            logger.error(f"Error loading conversation data from {path}: {e}")
    
    def load_text_corpus(self, path: str, chunk_size: int = 1000, stride: int = 500,
                         max_chunks: int = 100) -> None:
        """
        Load raw text corpus and create input-output pairs.
        
        Args:
            path: Path to the text file
            chunk_size: Size of each chunk (in characters)
            stride: Stride between chunks
            max_chunks: Maximum number of chunks to extract
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Create chunks with overlap
            chunks = []
            pos = 0
            count = 0
            
            while pos < len(text) - chunk_size and count < max_chunks:
                chunk = text[pos:pos + chunk_size]
                chunks.append(chunk)
                pos += stride
                count += 1
            
            # Create input-output pairs
            for i in range(len(chunks) - 1):
                input_text = chunks[i]
                output_text = chunks[i + 1]
                
                self.training_examples.append({
                    "input": input_text,
                    "output": output_text,
                    "source": os.path.basename(path)
                })
                
                # Update vocabulary
                words = re.findall(r'\b\w+\b', input_text.lower())
                self.vocabulary.update(words)
            
            logger.info(f"Created {len(chunks) - 1} text chunks from {path}")
        
        except Exception as e:
            logger.error(f"Error loading text corpus from {path}: {e}")
    
    def create_word_embeddings(self, embedding_dim: int = 50) -> None:
        """
        Create simple word embeddings based on word co-occurrence.
        
        Args:
            embedding_dim: Dimensionality of the embeddings
        """
        # Create word indices
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Initialize co-occurrence matrix
        n_words = len(self.vocabulary)
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        # Fill co-occurrence matrix
        window_size = 5
        for example in self.training_examples:
            words = re.findall(r'\b\w+\b', example["input"].lower())
            for i, word in enumerate(words):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                for j in range(start, end):
                    if i != j and words[j] in self.word_to_idx:
                        co_occurrence[word][words[j]] += 1
        
        # Convert to simple embeddings using random projections
        import random
        random.seed(42)  # For reproducibility
        
        self.word_embeddings = {}
        for word in self.vocabulary:
            # Start with random vector
            embedding = [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
            
            # Add co-occurrence information (if available)
            if word in co_occurrence:
                for co_word, count in co_occurrence[word].items():
                    random_dim = hash(co_word) % embedding_dim
                    embedding[random_dim] += math.log(count + 1)
            
            # Normalize
            magnitude = math.sqrt(sum(x*x for x in embedding))
            if magnitude > 0:
                embedding = [x/magnitude for x in embedding]
            
            self.word_embeddings[word] = embedding
        
        logger.info(f"Created {embedding_dim}-dimensional word embeddings for {len(self.word_embeddings)} words")
    
    def process_data(self, save_results: bool = True) -> Dict:
        """
        Process all loaded data and create training resources.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with processed data
        """
        # Create word embeddings
        self.create_word_embeddings()
        
        # Create template patterns
        patterns = self._extract_patterns()
        
        result = {
            "training_examples": self.training_examples,
            "vocabulary": list(self.vocabulary),
            "word_embeddings": self.word_embeddings,
            "patterns": patterns
        }
        
        if save_results:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved processed data to {self.output_path}")
        
        return result
    
    def _extract_patterns(self) -> List[Dict]:
        """
        Extract common patterns from training examples.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Group examples by similar inputs
        similarity_groups = {}
        for i, example in enumerate(self.training_examples):
            input_text = example["input"].lower()
            matched = False
            
            for key in list(similarity_groups.keys()):
                if self._calculate_similarity(input_text, key) > 0.7:
                    similarity_groups[key].append(i)
                    matched = True
                    break
            
            if not matched:
                similarity_groups[input_text] = [i]
        
        # Extract patterns from groups
        for key, indices in similarity_groups.items():
            if len(indices) > 1:
                examples = [self.training_examples[i] for i in indices]
                pattern = self._create_pattern(examples)
                if pattern:
                    patterns.append(pattern)
        
        logger.info(f"Extracted {len(patterns)} common patterns")
        return patterns
    
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
    
    def _create_pattern(self, examples: List[Dict]) -> Optional[Dict]:
        """
        Create a pattern from similar examples.
        
        Args:
            examples: List of similar examples
            
        Returns:
            Pattern dictionary or None if pattern creation failed
        """
        inputs = [ex["input"].lower() for ex in examples]
        outputs = [ex["output"] for ex in examples]
        
        # Extract common words and variable parts
        words_sets = [set(re.findall(r'\b\w+\b', inp)) for inp in inputs]
        common_words = set.intersection(*words_sets) if words_sets else set()
        
        if not common_words:
            return None
        
        # Create input pattern regex
        input_pattern = "|".join(re.escape(w) for w in common_words)
        
        # Extract response templates
        return {
            "input_pattern": input_pattern,
            "common_words": list(common_words),
            "response_templates": outputs
        }


class SimpleTrainer:
    """
    A simple trainer for the GAKR chatbot that doesn't require heavy ML libraries.
    """
    
    def __init__(self, data_processor: EnhancedDataProcessor):
        """
        Initialize the trainer.
        
        Args:
            data_processor: EnhancedDataProcessor instance with processed data
        """
        self.data_processor = data_processor
        self.model = None
    
    def train(self, model_path: str = "./models/trained/simple_model.pkl") -> None:
        """
        Train a simple model using the processed data.
        
        Args:
            model_path: Path to save the trained model
        """
        # Process data if not already processed
        if not self.data_processor.word_embeddings:
            processed_data = self.data_processor.process_data(save_results=True)
        else:
            processed_data = {
                "training_examples": self.data_processor.training_examples,
                "vocabulary": list(self.data_processor.vocabulary),
                "word_embeddings": self.data_processor.word_embeddings,
                "patterns": self.data_processor._extract_patterns()
            }
        
        # Create a simple model
        model = {
            "word_embeddings": processed_data["word_embeddings"],
            "patterns": processed_data["patterns"],
            "examples": processed_data["training_examples"],
            "metadata": {
                "version": "1.0",
                "timestamp": time.time(),
                "vocabulary_size": len(processed_data["vocabulary"]),
                "embedding_dim": len(next(iter(processed_data["word_embeddings"].values()))) if processed_data["word_embeddings"] else 0,
                "training_examples": len(processed_data["training_examples"])
            }
        }
        
        self.model = model
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Trained model saved to {model_path}")


def download_dataset(dataset_name: str, output_dir: str = "./datasets") -> str:
    """
    Download a dataset from a URL.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.txt")
    
    # For demonstration, we'll just use our existing training data
    # In a real implementation, this would download the dataset from a URL
    if dataset_name == "custom":
        return "training_data.json"
    
    logger.warning(f"Dataset {dataset_name} not available for download")
    return None


def main():
    """
    Main function to train the GAKR chatbot model.
    """
    logger.info("Starting GAKR Enhanced Training")
    
    # Check for command-line arguments
    import sys
    training_file = "training_data.json"
    if len(sys.argv) > 1:
        training_file = sys.argv[1]
        logger.info(f"Using training file: {training_file}")
    
    # Create data processor
    processor = EnhancedDataProcessor()
    
    # Load custom data
    processor.load_custom_data(training_file)
    
    # Create and process dataset
    processor.process_data()
    
    # Train model
    trainer = SimpleTrainer(processor)
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()