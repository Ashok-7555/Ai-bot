"""
GAKR AI Chatbot - Chat API Routes
This module provides API endpoints for the chat functionality.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from flask import Blueprint, jsonify, request, current_app, session
from flask_login import current_user

from app.models import User, UserProfile, Conversation, Message
from app.database import db
# Import automl_manager instead of sentiment_analyzer directly
# from app.ai_engine.sentiment_analyzer import analyze_sentiment
from app.ai_engine.automl_manager import get_automl_prediction

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprint
chat_api = Blueprint('chat_api', __name__)

@chat_api.route('/api/gakr', methods=['POST'])
def gakr_api():
    """API endpoint for GAKR AI chat interactions."""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Message is required.'}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    
    # Get complexity level from request
    complexity_level = data.get('complexity', 3)  # Default to medium if not provided
    
    # Validate complexity level
    if not isinstance(complexity_level, int) or complexity_level < 1 or complexity_level > 5:
        complexity_level = 3  # Reset to default if invalid
    
    # Get current user ID for authenticated users
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Convert client-side history to proper format if provided
    conversation_history = []
    if history and isinstance(history, list):
        for entry in history:
            if 'sender' in entry and 'message' in entry:
                is_user = entry['sender'] == 'user'
                conversation_history.append({
                    'content': entry['message'],
                    'is_user': is_user
                })
    
    # Store the most recent complexity preference in session
    session['complexity_level'] = complexity_level
    
    # Process the message using GAKR AI
    try:
        # Try using the AutoML model for sentiment analysis
        sentiment_result = None
        try:
            sentiment_result = get_automl_prediction('sentiment_analysis', user_message)
            if sentiment_result and sentiment_result.get('status') == 'success':
                sentiment = sentiment_result.get('prediction', '')
                logger.info(f"AutoML detected sentiment: {sentiment}")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
        
        # Adjust response based on complexity level
        conversation_input = {
            'message': user_message,
            'history': conversation_history[-4:] if len(conversation_history) > 2 else [],
            'complexity': complexity_level
        }
        
        # Try to get a model-generated response
        response_text = generate_response(user_message, conversation_history, complexity_level)
        
        # If we couldn't get a response, create a fallback
        if not response_text:
            logger.warning("Could not generate response using models, using fallback")
            response_text = generate_fallback_response(complexity_level)
        
        # Create a new conversation if needed for authenticated users
        if user_id:
            save_conversation(user_id, user_message, response_text)
        
        # Return the response
        return jsonify({
            'response': response_text,
            'sentiment': sentiment_result.get('prediction') if sentiment_result else None,
            'complexity': complexity_level
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your message.',
            'response': 'I apologize, but I encountered an issue while processing your message. Please try again.'
        }), 500


def generate_response(user_message: str, conversation_history: List[Dict], complexity_level: int) -> Optional[str]:
    """
    Generate a response based on user message and conversation history.
    
    Args:
        user_message: User's input message
        conversation_history: Previous conversation turns
        complexity_level: Level of complexity for the response (1-5)
        
    Returns:
        Generated response text or None if error occurs
    """
    try:
        # Adjust based on complexity level - simple responses
        if complexity_level <= 2:
            # Keep responses simpler for lower complexity
            simple_responses = [
                f"I understand you're talking about {extract_topic(user_message)}. That's interesting!",
                f"Thanks for your message about {extract_topic(user_message)}. I'll try to keep my answer simple.",
                f"I see you're interested in {extract_topic(user_message)}. Here's a straightforward answer.",
                f"Let me give you a simple explanation about {extract_topic(user_message)}.",
            ]
            return random.choice(simple_responses)
            
        # Medium complexity (default)
        elif complexity_level == 3:
            # Standard response generation
            standard_responses = [
                f"Thank you for asking about {extract_topic(user_message)}. Here's what I know about it.",
                f"That's a good question about {extract_topic(user_message)}. Let me share my thoughts.",
                f"I've processed your question about {extract_topic(user_message)}. Here's my response.",
                f"Regarding {extract_topic(user_message)}, I can provide you with some information.",
            ]
            return random.choice(standard_responses)
            
        # Higher complexity
        else:
            # More detailed responses for higher complexity
            complex_responses = [
                f"I appreciate your inquiry about {extract_topic(user_message)}. Let me provide a comprehensive analysis.",
                f"Your question about {extract_topic(user_message)} is quite interesting. I'll give you a detailed explanation.",
                f"Regarding {extract_topic(user_message)}, there are several important aspects to consider in depth.",
                f"Let me elaborate on {extract_topic(user_message)} with a thorough and nuanced perspective.",
            ]
            return random.choice(complex_responses)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None


def generate_fallback_response(complexity_level: int) -> str:
    """
    Generate a fallback response when the main response generation fails.
    
    Args:
        complexity_level: Level of complexity for the response (1-5)
        
    Returns:
        Fallback response text
    """
    # Simple fallbacks
    if complexity_level <= 2:
        fallbacks = [
            "I understand what you're saying. Can you tell me more?",
            "That's interesting! What else would you like to know?",
            "I see. How can I help you with that?",
            "Thanks for sharing. Do you have any questions?",
        ]
    # Medium fallbacks
    elif complexity_level == 3:
        fallbacks = [
            "I appreciate your message. I'm processing that information. What specifically would you like to know?",
            "That's a good point. I'm thinking about the best way to respond. Can you elaborate?",
            "I understand what you're asking about. Could you provide more details?",
            "I'm analyzing your question. What other aspects of this topic interest you?",
        ]
    # Complex fallbacks
    else:
        fallbacks = [
            "Thank you for sharing that perspective. I'm considering multiple angles on this subject. Could you elaborate on what specific aspects you'd like me to address?",
            "That's a nuanced topic with several dimensions to explore. I'd like to ensure I'm addressing your specific interests. Could you clarify what you'd like me to focus on?",
            "I appreciate the complexity of your inquiry. To provide a comprehensive response, could you specify which aspects of this topic are most relevant to you?",
            "Your question touches on several interesting domains. To give you the most valuable analysis, could you indicate which direction you'd like me to explore further?",
        ]
    
    return random.choice(fallbacks)


def extract_topic(message: str) -> str:
    """
    Extract a simplified topic from the user message.
    
    Args:
        message: User's input message
        
    Returns:
        Simple topic string
    """
    # Simple keyword extraction
    keywords = ["AI", "chatbot", "learning", "data", "model", "language", 
                "computer", "technology", "science", "programming", 
                "information", "knowledge", "analysis", "system", "intelligence"]
    
    # Look for keywords in the message
    for keyword in keywords:
        if keyword.lower() in message.lower():
            return keyword.lower()
    
    # If no keywords found, use a generic topic
    return "this topic"


def save_conversation(user_id: int, user_message: str, bot_response: str) -> None:
    """
    Save the conversation to the database for authenticated users.
    
    Args:
        user_id: User's ID
        user_message: User's message
        bot_response: Bot's response
    """
    try:
        # Get or create current conversation
        conversation = Conversation.query.filter_by(
            user_id=user_id
        ).order_by(Conversation.updated_at.desc()).first()
        
        # Create a new conversation if none exists
        if not conversation:
            conversation = Conversation(
                user_id=user_id,
                title=user_message[:50] + "..." if len(user_message) > 50 else user_message
            )
            db.session.add(conversation)
            db.session.flush()
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        
        # Add user message
        user_msg = Message(
            conversation_id=conversation.id,
            content=user_message,
            is_user=True
        )
        db.session.add(user_msg)
        
        # Add bot response
        bot_msg = Message(
            conversation_id=conversation.id,
            content=bot_response,
            is_user=False
        )
        db.session.add(bot_msg)
        
        # Commit changes
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        db.session.rollback()