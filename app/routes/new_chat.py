import json
import random
import logging
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request, session
from flask_login import current_user, login_required

from app.models import Conversation, Message, User, UserProfile
from app.database import db

# Configure logger
logger = logging.getLogger(__name__)

# Import the AI modules
from app.ai_engine.sentiment_analyzer import analyze_text_sentiment
from app.ai_engine.model_manager import adjust_response_complexity, start_model_training
from app.ai_engine.automl_manager import get_automl_prediction

# Create a blueprint for the chat routes
chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

@chat_bp.route('/', methods=['GET'])
def chat():
    """Render the simplified chat interface."""
    messages = []
    
    # Get the user's conversation history if authenticated
    if current_user.is_authenticated:
        # Get most recent conversation or create a new one
        conversation = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).first()
        
        if conversation:
            # Get messages for this conversation
            messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at).all()
    
    # Return the chat interface template with any existing messages
    return render_template('chat.html', messages=messages)

@chat_bp.route('/api/gakr', methods=['POST'])
def gakr_api():
    """API endpoint for GAKR AI chat interactions."""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Message is required.'}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    
    # Get user ID for authenticated users
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Default complexity level
    complexity_level = 3
    
    # Process the message using GAKR AI
    try:
        # Try using the AutoML model for sentiment analysis
        sentiment_result = None
        try:
            sentiment_result = get_automl_prediction('sentiment_analysis', user_message)
            if sentiment_result.get('status') == 'success':
                sentiment = sentiment_result.get('prediction', '')
                logger.info(f"AutoML detected sentiment: {sentiment}")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
        
        # Try using the AutoML model for conversation generation
        automl_response = None
        try:
            conversation_input = {
                'message': user_message,
                'history': history[-4:] if len(history) > 2 else []  # Last 2 exchanges
            }
            conversation_result = get_automl_prediction('conversation_generation', conversation_input)
            if conversation_result.get('status') == 'success' and conversation_result.get('response'):
                automl_response = conversation_result.get('response')
                logger.info(f"AutoML generated response: {automl_response}")
        except Exception as e:
            logger.error(f"Error in conversation generation: {str(e)}")
        
        # Use the AutoML response or fallback to rule-based response
        if automl_response and len(automl_response) > 5:
            base_response = automl_response
        else:
            # Rule-based fallback responses
            if not user_message or len(user_message.strip()) < 3:
                base_response = "Could you please provide more information so I can assist you properly?"
            elif any(keyword in user_message.lower() for keyword in ["hello", "hi", "hey", "greetings"]):
                base_response = "Hello! How can I assist you today with GAKR AI?"
            elif any(keyword in user_message.lower() for keyword in ["thank", "thanks", "appreciate"]):
                base_response = "You're welcome! Is there anything else I can help you with?"
            elif any(keyword in user_message.lower() for keyword in ["bye", "goodbye", "see you"]):
                base_response = "Goodbye! Feel free to return whenever you have more questions."
            elif "?" in user_message:
                # For questions, use a knowledge base lookup
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                knowledge_base = [
                    {"q": "what can you do", "a": "I can answer questions, provide information, and assist with various tasks using natural language processing and machine learning."},
                    {"q": "how do you work", "a": "I process text using advanced NLP algorithms and AutoML techniques to understand meaning and generate relevant responses."},
                    {"q": "who made you", "a": "I was created by a team of developers as an AI assistant that runs completely locally without external APIs."},
                    {"q": "what is artificial intelligence", "a": "Artificial Intelligence (AI) refers to systems designed to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and understanding natural language."},
                    {"q": "what is machine learning", "a": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."},
                    {"q": "what is nlp", "a": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language."},
                    {"q": "what is gakr", "a": "GAKR is an AI assistant designed to process natural language, answer questions, and provide helpful information without relying on external APIs."}
                ]
                
                questions = [item["q"] for item in knowledge_base]
                answers = [item["a"] for item in knowledge_base]
                
                vectorizer = TfidfVectorizer().fit(questions + [user_message])
                question_vectors = vectorizer.transform(questions)
                query_vector = vectorizer.transform([user_message])
                
                similarities = cosine_similarity(query_vector, question_vectors)
                most_similar_idx = np.argmax(similarities)
                similarity_score = similarities[0][most_similar_idx]
                
                if similarity_score > 0.3:
                    base_response = answers[most_similar_idx]
                else:
                    base_response = "That's an interesting question. I don't have a specific answer, but I'm designed to help with various tasks through natural language understanding."
            else:
                # For statements, provide a general response
                base_response = "I understand what you're saying. GAKR AI is here to assist you with information and answers to your questions."
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        base_response = "I'm having trouble processing your message. Please try again."
    
    # Adjust response complexity
    response_text = adjust_response_complexity(base_response, complexity_level)
    
    # Analyze sentiment
    user_sentiment = analyze_text_sentiment(user_message)
    response_sentiment = analyze_text_sentiment(response_text)
    
    # Store the conversation and messages if user is authenticated
    conversation_id = None
    if user_id:
        # Get or create conversation
        conversation = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.updated_at.desc()).first()
        
        if not conversation or (datetime.utcnow() - conversation.updated_at).total_seconds() > 3600:  # 1 hour timeout
            conversation = Conversation(
                user_id=user_id,
                title=user_message[:50] + "..." if len(user_message) > 50 else user_message
            )
            db.session.add(conversation)
            db.session.commit()
        
        conversation_id = conversation.id
        
        # Add user message
        user_msg = Message(
            conversation_id=conversation_id,
            content=user_message,
            is_user=True
        )
        db.session.add(user_msg)
        
        # Add AI response
        ai_msg = Message(
            conversation_id=conversation_id,
            content=response_text,
            is_user=False
        )
        db.session.add(ai_msg)
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        db.session.commit()
    
    # Return response as JSON
    return jsonify({
        'response': response_text,
        'conversation_id': conversation_id,
        'user_sentiment': {
            'sentiment': user_sentiment['sentiment'],
            'score': user_sentiment['compound']
        },
        'response_sentiment': {
            'sentiment': response_sentiment['sentiment'],
            'score': response_sentiment['compound']
        }
    })