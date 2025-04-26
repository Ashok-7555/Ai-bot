"""
GAKR AI - Sentiment Analysis Module
This module analyzes text sentiment using NLTK and spaCy.
"""

import os
import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker

# Ensure NLTK resources are downloaded
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyze sentiment in text messages using NLTK's VADER.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.spell_checker = SpellChecker()
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            # Fallback to dummy analyzer if initialization fails
            self.sia = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text message.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment scores and classification
        """
        if not text or not self.sia:
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'sentiment': 'neutral',
                'confidence': 0
            }
        
        # Clean text for analysis
        cleaned_text = self._clean_text(text)
        
        # Get sentiment scores
        scores = self.sia.polarity_scores(cleaned_text)
        
        # Determine overall sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on the absolute value of the compound score
        confidence = min(abs(compound) * 2, 1.0)  # Scale to 0-1
        
        # Add classification and confidence to scores
        scores['sentiment'] = sentiment
        scores['confidence'] = confidence
        
        return scores
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters except punctuation important for sentiment
        text = re.sub(r'[^\w\s!?.,;:]', '', text)
        
        # Check spelling for long words (to avoid correcting names and abbreviations)
        words = text.split()
        corrected_words = []
        for word in words:
            if len(word) > 3:
                corrected_word = self.spell_checker.correction(word)
                if corrected_word:
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def get_emoji_for_sentiment(self, sentiment: str) -> str:
        """
        Get an emoji representing the given sentiment.
        
        Args:
            sentiment: The sentiment label
            
        Returns:
            Emoji character
        """
        emoji_map = {
            'positive': '😊',
            'negative': '😔',
            'neutral': '😐',
            'very_positive': '😄',
            'very_negative': '😡'
        }
        return emoji_map.get(sentiment, '😐')

    def get_color_for_sentiment(self, sentiment: str) -> str:
        """
        Get a color code representing the given sentiment.
        
        Args:
            sentiment: The sentiment label
            
        Returns:
            Hex color code
        """
        color_map = {
            'positive': '#28a745',  # Green
            'negative': '#dc3545',  # Red
            'neutral': '#6c757d',   # Gray
            'very_positive': '#198754', # Dark green
            'very_negative': '#b71c1c'  # Dark red
        }
        return color_map.get(sentiment, '#6c757d')
    
    def get_detailed_sentiment(self, compound_score: float) -> str:
        """
        Get a more detailed sentiment classification based on the compound score.
        
        Args:
            compound_score: VADER compound sentiment score
            
        Returns:
            Detailed sentiment label
        """
        if compound_score >= 0.5:
            return 'very_positive'
        elif 0.05 <= compound_score < 0.5:
            return 'positive'
        elif -0.05 < compound_score < 0.05:
            return 'neutral'
        elif -0.5 < compound_score <= -0.05:
            return 'negative'
        else:
            return 'very_negative'

class ConversationAnalyzer:
    """
    Analyze conversation patterns and metrics.
    """
    
    def __init__(self):
        """Initialize the conversation analyzer."""
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a conversation based on a list of messages.
        
        Args:
            messages: List of message dictionaries with 'content' and 'is_user' keys
            
        Returns:
            Dictionary with conversation metrics and insights
        """
        if not messages:
            return {
                'messages_count': 0,
                'avg_sentiment': 'neutral',
                'sentiment_trend': 'neutral',
                'avg_message_length': 0,
                'topic_keywords': [],
                'user_engagement': 0
            }
        
        # Extract message contents
        user_messages = [msg['content'] for msg in messages if msg.get('is_user', False)]
        ai_messages = [msg['content'] for msg in messages if not msg.get('is_user', False)]
        
        # Calculate sentiment for all messages
        sentiments = [self.sentiment_analyzer.analyze(msg['content']) for msg in messages]
        
        # Calculate average sentiment compound score
        avg_compound = np.mean([s['compound'] for s in sentiments]) if sentiments else 0
        
        # Get sentiment trend (comparing first and second half of conversation)
        mid_point = len(sentiments) // 2
        if mid_point > 0 and len(sentiments) > 2:
            first_half = np.mean([s['compound'] for s in sentiments[:mid_point]])
            second_half = np.mean([s['compound'] for s in sentiments[mid_point:]])
            trend_diff = second_half - first_half
            
            if trend_diff > 0.2:
                sentiment_trend = 'improving'
            elif trend_diff < -0.2:
                sentiment_trend = 'deteriorating'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'neutral'
        
        # Calculate average message length
        msg_lengths = [len(msg['content'].split()) for msg in messages]
        avg_length = np.mean(msg_lengths) if msg_lengths else 0
        
        # Estimate user engagement (ratio of user message length to AI message length)
        avg_user_length = np.mean([len(m.split()) for m in user_messages]) if user_messages else 0
        avg_ai_length = np.mean([len(m.split()) for m in ai_messages]) if ai_messages else 1
        engagement = min(avg_user_length / max(avg_ai_length, 1), 1.0)
        
        # Get detailed sentiment classification
        detailed_sentiment = self.sentiment_analyzer.get_detailed_sentiment(avg_compound)
        
        return {
            'messages_count': len(messages),
            'avg_sentiment': detailed_sentiment,
            'sentiment_score': avg_compound,
            'sentiment_trend': sentiment_trend,
            'avg_message_length': avg_length,
            'user_engagement': engagement,
            'sentiment_color': self.sentiment_analyzer.get_color_for_sentiment(detailed_sentiment),
            'sentiment_emoji': self.sentiment_analyzer.get_emoji_for_sentiment(detailed_sentiment)
        }

# Create singleton instances for convenience
sentiment_analyzer = SentimentAnalyzer()
conversation_analyzer = ConversationAnalyzer()

def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of a text message.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    return sentiment_analyzer.analyze(text)

def analyze_conversation_metrics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze conversation metrics and patterns.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with conversation analysis results
    """
    return conversation_analyzer.analyze_conversation(messages)