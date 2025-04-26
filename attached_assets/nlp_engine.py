"""
GAKR AI - NLP Engine
This module provides natural language processing capabilities for the GAKR AI chatbot.
"""

import re
import math
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPEngine:
    """
    Natural Language Processing engine for text analysis and manipulation.
    """
    
    def __init__(self):
        """Initialize the NLP engine with default configurations."""
        # Compile regex patterns for efficiency
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split text by sentence-ending punctuation
        sentences = self.sentence_split_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def calculate_similarity(self, text1: str, text2: str, method: str = "cosine") -> float:
        """
        Calculate text similarity between two strings.
        
        Args:
            text1: First string
            text2: Second string
            method: Similarity method ('cosine', 'jaccard', or 'simple')
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Lowercase and tokenize
        tokens1 = re.findall(r'\b\w+\b', text1.lower())
        tokens2 = re.findall(r'\b\w+\b', text2.lower())
        
        if method == "simple":
            # Simple word overlap
            common_words = set(tokens1).intersection(set(tokens2))
            return len(common_words) / max(len(set(tokens1)), len(set(tokens2)), 1)
            
        elif method == "jaccard":
            # Jaccard similarity: intersection / union
            set1 = set(tokens1)
            set2 = set(tokens2)
            
            if not set1 or not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union
            
        else:  # Default: cosine similarity
            # Count word frequencies
            vec1 = Counter(tokens1)
            vec2 = Counter(tokens2)
            
            # Find all unique words
            all_words = set(vec1.keys()).union(set(vec2.keys()))
            
            # Calculate dot product
            dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(vec1.get(word, 0) ** 2 for word in all_words))
            magnitude2 = math.sqrt(sum(vec2.get(word, 0) ** 2 for word in all_words))
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
                
            return dot_product / (magnitude1 * magnitude2)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract basic named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {
            "names": [],
            "locations": [],
            "organizations": [],
            "dates": [],
            "numbers": [],
            "email": [],
            "url": []
        }
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities["email"] = emails
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
        entities["url"] = urls
        
        # Extract dates (simple patterns)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
        dates += re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE)
        entities["dates"] = dates
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities["numbers"] = numbers
        
        # Return extracted entities
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform simple rule-based sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Lists of positive and negative words
        positive_words = [
            "good", "great", "excellent", "amazing", "love", "happy", "wonderful", 
            "awesome", "fantastic", "nice", "pleased", "glad", "joy", "exciting", 
            "brilliant", "terrific", "perfect", "impressive", "beautiful", "enjoy",
            "like", "best", "positive", "amazing", "recommended", "helpful", "useful"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "hate", "poor", "sad", "horrible", "dislike", 
            "disappointed", "upset", "angry", "frustrating", "annoying", "useless", 
            "stupid", "boring", "worse", "worst", "not satisfied", "negative", "problem",
            "issue", "difficult", "fail", "failure", "trouble", "wrong", "broken", "error"
        ]
        
        # Convert to lowercase for word matching
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', text_lower))
        negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', text_lower))
        
        # Check for negations which can flip sentiment
        negations = ["not", "no", "never", "neither", "nor", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't"]
        negation_count = sum(1 for word in negations if re.search(r'\b' + word + r'\b', text_lower))
        
        # Adjust scores based on negations (simplified approach)
        if negation_count > 0:
            temp = positive_count
            positive_count = negative_count
            negative_count = temp
        
        # Determine sentiment
        sentiment = "neutral"
        score = 0.5  # Default neutral score
        
        if positive_count > negative_count:
            sentiment = "positive"
            # Calculate score between 0.5 and 1 based on positive word count
            score = 0.5 + (0.5 * min(1, positive_count / max(1, len(text.split()) / 10)))
        elif negative_count > positive_count:
            sentiment = "negative"
            # Calculate score between 0 and 0.5 based on negative word count
            score = 0.5 - (0.5 * min(1, negative_count / max(1, len(text.split()) / 10)))
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "intensity": abs(score - 0.5) * 2  # Intensity from 0 to 1
        }
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate a simple extractive summary from text.
        
        Args:
            text: Input text
            num_sentences: Number of sentences to include in summary
            
        Returns:
            Summarized text
        """
        if not text:
            return ""
        
        # Split text into sentences
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
            
        # Simple extractive summarization based on sentence position and length
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Score based on position (beginning and end sentences are more important)
            position_score = 1.0
            if i < len(sentences) / 3:  # First third
                position_score = 1.5
            elif i >= 2 * len(sentences) / 3:  # Last third
                position_score = 1.2
                
            # Score based on sentence length (prefer medium-length sentences)
            words = len(sentence.split())
            length_score = 1.0
            if 5 <= words <= 20:  # Medium length
                length_score = 1.3
            elif words < 5:  # Too short
                length_score = 0.7
            elif words > 30:  # Too long
                length_score = 0.9
                
            # Final score is a combination of factors
            final_score = position_score * length_score
            
            scored_sentences.append((sentence, final_score))
            
        # Sort sentences by score (higher is better)
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Take top sentences
        top_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
        
        # Sort sentences by original order
        ordered_sentences = [sentence for sentence in sentences if sentence in top_sentences]
        
        # Join sentences
        return " ".join(ordered_sentences)
    
    def categorize_text(self, text: str) -> List[str]:
        """
        Categorize text into predefined topics.
        
        Args:
            text: Input text
            
        Returns:
            List of relevant topic categories
        """
        # Define category keywords
        categories = {
            "technology": ["computer", "technology", "software", "hardware", "programming", 
                          "code", "app", "algorithm", "data", "internet", "online", "digital",
                          "ai", "artificial intelligence", "machine learning"],
            "business": ["business", "company", "startup", "entrepreneur", "market", "finance",
                        "investment", "stock", "profit", "industry", "commercial", "economy"],
            "health": ["health", "medical", "doctor", "medicine", "disease", "illness", "treatment",
                      "hospital", "wellness", "fitness", "diet", "nutrition", "exercise"],
            "education": ["education", "school", "university", "college", "student", "teacher",
                         "learning", "course", "study", "academic", "research", "knowledge"],
            "entertainment": ["movie", "film", "music", "song", "game", "play", "entertainment",
                             "show", "book", "novel", "story", "actor", "performer", "stage"],
            "science": ["science", "scientific", "experiment", "physics", "chemistry", "biology",
                       "astronomy", "research", "theory", "hypothesis", "laboratory", "discovery"],
            "sports": ["sport", "athlete", "team", "game", "competition", "player", "championship",
                      "win", "lose", "score", "match", "tournament", "stadium", "ball"]
        }
        
        text_lower = text.lower()
        matched_categories = []
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    matched_categories.append(category)
                    break
        
        return matched_categories
    
    def suggest_questions(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        Suggest follow-up questions based on text content and conversation history.
        
        Args:
            text: Input text
            conversation_history: Optional conversation history
            
        Returns:
            List of suggested follow-up questions
        """
        suggestions = []
        
        # Extract main topics from text
        categories = self.categorize_text(text)
        entities = self.extract_entities(text)
        
        # Generate topic-based questions
        if "technology" in categories:
            suggestions.append("How does this technology work?")
            suggestions.append("What are the most recent advancements in this field?")
            
        if "business" in categories:
            suggestions.append("What are the market implications of this?")
            suggestions.append("How does this affect the industry landscape?")
            
        if "health" in categories:
            suggestions.append("What are the health benefits or risks associated with this?")
            suggestions.append("Is there research supporting these claims?")
            
        if "education" in categories:
            suggestions.append("How is this taught or implemented in educational settings?")
            suggestions.append("What resources are available to learn more about this topic?")
        
        # Generate entity-based questions
        if entities["names"]:
            suggestions.append(f"Can you tell me more about {random.choice(entities['names'])}?")
            
        if entities["organizations"]:
            suggestions.append(f"What is {random.choice(entities['organizations'])} known for?")
            
        if entities["locations"]:
            suggestions.append(f"What's significant about {random.choice(entities['locations'])}?")
        
        # Generate general follow-up questions
        general_questions = [
            "Could you elaborate more on this topic?",
            "What are the main benefits and drawbacks?",
            "How has this evolved over time?",
            "What are the future trends in this area?",
            "Are there alternative approaches to consider?"
        ]
        
        # Add some general questions
        suggestions.extend(random.sample(general_questions, min(2, len(general_questions))))
        
        # Limit the number of suggestions
        return suggestions[:3]

# Create a singleton instance
nlp_engine = NLPEngine()

def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment analysis results
    """
    return nlp_engine.analyze_sentiment(text)

def calculate_text_similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method
        
    Returns:
        Similarity score
    """
    return nlp_engine.calculate_similarity(text1, text2, method)

def extract_text_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract entities from text.
    
    Args:
        text: Input text
        
    Returns:
        Extracted entities
    """
    return nlp_engine.extract_entities(text)

def summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text.
    
    Args:
        text: Input text
        num_sentences: Number of sentences in summary
        
    Returns:
        Summarized text
    """
    return nlp_engine.summarize_text(text, num_sentences)