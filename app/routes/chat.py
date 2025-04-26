import json
import random
import logging
from datetime import datetime

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for, session
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, NumberRange, Optional

from app.models import Conversation, Message, User, UserProfile
from app.database import db

# Configure logger
logger = logging.getLogger(__name__)

# Import the AI modules
from app.ai_engine.sentiment_analyzer import analyze_text_sentiment, analyze_conversation_metrics
from app.ai_engine.model_manager import adjust_response_complexity, get_complexity_levels, start_model_training, get_training_status

# Define the chat form
class ChatForm(FlaskForm):
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Send')

# Define the model settings form
class ModelSettingsForm(FlaskForm):
    complexity_level = SelectField('Model Complexity', 
                                  choices=[(1, 'Simple'), (2, 'Basic'), (3, 'Standard'), 
                                           (4, 'Detailed'), (5, 'Advanced')],
                                  validators=[DataRequired()],
                                  coerce=int,
                                  default=3)
    
    theme = SelectField('Interface Theme', 
                       choices=[('light', 'Light'), ('dark', 'Dark'), ('blue', 'Blue'), 
                                ('green', 'Green'), ('purple', 'Purple')],
                       default='light')
    
    show_sentiment = BooleanField('Show Sentiment Analysis', default=True)
    
    auto_train = BooleanField('Enable Auto-Training', default=False)
    
    submit = SubmitField('Save Settings')

# Create a blueprint for the chat routes
chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

# Sample responses for demonstration
SAMPLE_RESPONSES = [
    "I understand your question. Based on my analysis, I'd recommend...",
    "That's an interesting point. There are several ways to approach this...",
    "From my understanding, the key factors to consider are...",
    "I've analyzed your request, and here's what I found...",
    "Let me process that information. The most relevant solution would be...",
]

@chat_bp.route('/', methods=['GET'])
def chat():
    """Render the chat interface."""
    form = ChatForm()
    conversation_id = request.args.get('conversation_id')
    conversation = None
    messages = []
    conversations = []
    
    # Check for guest users
    if not current_user.is_authenticated:
        # Allow guest access but set a session limit
        guest_conversation_count = request.cookies.get('guest_conversation_count', 0)
        if int(guest_conversation_count) >= 5:
            flash('You have reached the maximum number of guest conversations. Please register to continue.', 'info')
            return redirect(url_for('auth.register'))
        
        is_guest = True
        return render_template('chat.html', is_guest=is_guest, guest_conversation_count=int(guest_conversation_count), form=form)
    
    # For authenticated users
    # Get user's conversations for the sidebar
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).limit(10).all()
    
    # Load a specific conversation if requested
    if conversation_id:
        conversation = Conversation.query.get(conversation_id)
        # Check that the conversation exists and belongs to the user
        if conversation and conversation.user_id == current_user.id:
            messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at).all()
        else:
            flash('Conversation not found or access denied.', 'warning')
            # Redirect to the main chat page without a conversation ID
            return redirect(url_for('chat.chat'))
    
    # Always pass conversations to the template for the sidebar
    return render_template('chat.html', conversation=conversation, conversations=conversations, messages=messages, form=form)

@chat_bp.route('/history', methods=['GET'])
@login_required
def history():
    """Display conversation history."""
    # Get sort parameter from URL
    sort = request.args.get('sort', 'newest')
    
    # Query for user's conversations
    query = Conversation.query.filter_by(user_id=current_user.id)
    
    # Apply sorting
    if sort == 'newest':
        query = query.order_by(Conversation.updated_at.desc())
    elif sort == 'oldest':
        query = query.order_by(Conversation.created_at.asc())
    elif sort == 'title':
        query = query.order_by(Conversation.title.asc())
    
    conversations = query.all()
    
    return render_template('history.html', conversations=conversations)

@chat_bp.route('/conversation/<int:conversation_id>', methods=['GET'])
@login_required
def conversation_detail(conversation_id):
    """Display a specific conversation."""
    form = ChatForm()
    conversation = Conversation.query.get_or_404(conversation_id)
    
    # Check if the user has access to this conversation
    if conversation.user_id != current_user.id:
        abort(403)  # Forbidden
    
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at).all()
    return render_template('conversation_detail.html', conversation=conversation, messages=messages, form=form)

@chat_bp.route('/api/gakr', methods=['POST'])
def gakr_api():
    """API endpoint for GAKR AI chat interactions."""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request. Message is required.'}), 400
        
        user_message = data['message']
        history = data.get('history', [])

        # Get sentiment analysis
        sentiment_result = analyze_text_sentiment(user_message)
        sentiment = sentiment_result.get('sentiment', 'neutral')
        
        # Use core NLP engine for response generation
        from core.nlp_engine import NLPEngine
        nlp_engine = NLPEngine()
        response, confidence = nlp_engine.generate_response(user_message, history)
        
        return jsonify({
            'response': response,
            'conversation_id': None,
            'user_sentiment': {
                'sentiment': sentiment,
                'score': sentiment_result.get('compound', 0),
                'emoji': '😊' if sentiment == 'positive' else '😐' if sentiment == 'neutral' else '😔',
                'color': '#28a745' if sentiment == 'positive' else '#6c757d' if sentiment == 'neutral' else '#dc3545'
            },
            'response_sentiment': {
                'sentiment': 'neutral',
                'score': 0,
                'emoji': '😊',
                'color': '#6c757d'
            }
        })
    except Exception as e:
        logger.error(f"Error in GAKR API: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your message.',
            'response': 'I apologize, but I encountered an issue processing your message. Please try again.'
        }), 500
    
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
    
    # Get complexity level from user profile or session
    complexity_level = 3  # Default to standard complexity
    
    if current_user.is_authenticated:
        # Try to get from session first (most recently set)
        if 'complexity_level' in session:
            complexity_level = session['complexity_level']
        else:
            # Get from user profile
            profile = UserProfile.query.filter_by(user_id=current_user.id).first()
            if profile:
                # Map model preference to complexity level
                model_to_complexity = {
                    'simple': 1,
                    'basic': 2,
                    'standard': 3,
                    'advanced': 4,
                    'expert': 5
                }
                complexity_level = model_to_complexity.get(profile.preferred_model, 3)
    
    # Get or create conversation for authenticated users
    conversation_id = None
    if user_id:
        # Look for existing active conversation
        recent_conversation = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.updated_at.desc()).first()
        
        # Create new conversation if none exists or if the last one is old
        if not recent_conversation or (datetime.utcnow() - recent_conversation.updated_at).total_seconds() > 3600:  # 1 hour timeout
            # Create a new conversation
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            conversation = Conversation(
                user_id=user_id,
                title=title
            )
            db.session.add(conversation)
            db.session.commit()
            conversation_id = conversation.id
        else:
            conversation_id = recent_conversation.id
    
    # Process the message using GAKR AI with AutoML integration
    try:
        # First check if we can use the AutoML model for a response
        from app.ai_engine.automl_manager import get_automl_prediction
        automl_response = None
        
        # Try sentiment analysis first
        sentiment_result = get_automl_prediction('sentiment_analysis', user_message)
        if sentiment_result.get('status') == 'success':
            sentiment = sentiment_result.get('prediction', '')
            if isinstance(sentiment, str) and sentiment:
                logger.info(f"AutoML detected sentiment: {sentiment}")
        
        # Try entity recognition
        try:
            entity_result = get_automl_prediction('entity_recognition', user_message)
            if entity_result.get('status') == 'success' and entity_result.get('entities'):
                entities = entity_result.get('entities')
                logger.info(f"AutoML detected entities: {entities}")
        except Exception as e:
            logger.error(f"Error in entity recognition: {str(e)}")
        
        # Try conversation generation for response
        try:
            conversation_input = {
                'message': user_message,
                'history': history[-4:] if len(history) > 2 else []  # Last 2 exchanges (4 messages)
            }
            conversation_result = get_automl_prediction('conversation_generation', conversation_input)
            if conversation_result.get('status') == 'success' and conversation_result.get('response'):
                automl_response = conversation_result.get('response')
                logger.info(f"AutoML generated response: {automl_response}")
        except Exception as e:
            logger.error(f"Error in conversation generation: {str(e)}")
        
        # Use the AutoML response if available, otherwise fallback to rule-based response
        if automl_response and len(automl_response) > 5:
            base_response = automl_response
        else:
            # Rule-based response generation as fallback
            if not user_message or len(user_message.strip()) < 3:
                base_response = "Could you please provide more information so I can assist you properly?"
            elif any(keyword in user_message.lower() for keyword in ["hello", "hi", "hey", "greetings"]):
                base_response = "Hello! How can I assist you today with GAKR AI?"
            elif any(keyword in user_message.lower() for keyword in ["thank", "thanks", "appreciate"]):
                base_response = "You're welcome! Is there anything else I can help you with?"
            elif any(keyword in user_message.lower() for keyword in ["bye", "goodbye", "see you"]):
                base_response = "Goodbye! Feel free to return whenever you have more questions."
            elif "?" in user_message:
                # This is likely a question, try to provide a helpful response
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Sample knowledge base (would be more extensive in real implementation)
                knowledge_base = [
                    {"q": "what can you do", "a": "I can answer questions, provide information, and assist with various tasks using natural language processing and machine learning."},
                    {"q": "how do you work", "a": "I process text using advanced NLP algorithms and AutoML techniques to understand meaning and generate relevant responses."},
                    {"q": "who made you", "a": "I was created by a team of developers as an AI assistant that runs completely locally without external APIs."},
                    {"q": "what is artificial intelligence", "a": "Artificial Intelligence (AI) refers to systems designed to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and understanding natural language."},
                    {"q": "what is machine learning", "a": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."},
                    {"q": "what is nlp", "a": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language."},
                    {"q": "what is automl", "a": "AutoML (Automated Machine Learning) refers to tools and techniques that automate the process of applying machine learning to real-world problems, making AI more accessible."}
                ]
                
                # Create a simple search function
                questions = [item["q"] for item in knowledge_base]
                answers = [item["a"] for item in knowledge_base]
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer().fit(questions + [user_message])
                question_vectors = vectorizer.transform(questions)
                query_vector = vectorizer.transform([user_message])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, question_vectors)
                
                # Find the most similar question
                most_similar_idx = np.argmax(similarities)
                similarity_score = similarities[0][most_similar_idx]
                
                if similarity_score > 0.3:  # Threshold for relevance
                    base_response = answers[most_similar_idx]
                else:
                    # Generate a generic question response
                    base_response = "That's an interesting question. Based on my analysis, I would say it depends on several factors. Could you provide more context or specify your question further?"
            else:
                # For statements, try to provide a thoughtful response
                if len(user_message.split()) < 5:
                    base_response = "I understand. Could you tell me more about that?"
                else:
                    base_response = "I see what you're saying. Based on my understanding, this relates to concepts in " + \
                                  random.choice(["information processing", "knowledge representation", "decision making", 
                                              "pattern recognition", "language understanding", "cognitive systems"])
    
    except Exception as e:
        # Fallback in case of any error
        logger.error(f"Error generating response: {str(e)}")
        base_response = "I'm processing your message. Let me think about the best way to respond..."
    
    # Adjust response based on complexity level
    response_text = adjust_response_complexity(base_response, complexity_level)
    
    # Analyze sentiment of user message and response with null check
    if user_message:
        user_sentiment = analyze_text_sentiment(user_message)
    else:
        user_sentiment = {'sentiment': 'neutral', 'compound': 0}
        
    response_sentiment = analyze_text_sentiment(response_text)
    
    # Store the conversation and messages if user is authenticated
    if user_id and conversation_id:
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
        conversation = Conversation.query.get(conversation_id)
        if conversation:
            conversation.updated_at = datetime.utcnow()
            db.session.commit()
        
        # Store conversation in session
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        session_history = session['conversation_history']
        session_history.append({'sender': 'user', 'message': user_message})
        session_history.append({'sender': 'bot', 'message': response_text})
        
        # Limit history size in session
        if len(session_history) > 20:  # Keep last 10 exchanges (20 messages)
            session_history = session_history[-20:]
        
        session['conversation_history'] = session_history
    
    # Return json response with all data
    return jsonify({
        'response': response_text,
        'conversation_id': conversation_id,
        'user_sentiment': {
            'sentiment': user_sentiment['sentiment'],
            'score': user_sentiment['compound'],
            'emoji': user_sentiment.get('sentiment_emoji', ''),
            'color': user_sentiment.get('sentiment_color', '#6c757d')
        },
        'response_sentiment': {
            'sentiment': response_sentiment['sentiment'],
            'score': response_sentiment['compound'],
            'emoji': response_sentiment.get('sentiment_emoji', ''),
            'color': response_sentiment.get('sentiment_color', '#6c757d')
        }
    })

@chat_bp.route('/process', methods=['POST'])
def process_chat():
    """Process a chat message and return a response."""
    form = ChatForm()
    
    if form.validate_on_submit():
        message_text = form.message.data
        conversation_id = request.args.get('conversation_id')
        
        # Get complexity level from user profile or session
        complexity_level = 3  # Default to standard complexity
        
        if current_user.is_authenticated:
            # Try to get from session first (most recently set)
            if 'complexity_level' in session:
                complexity_level = session['complexity_level']
            else:
                # Get from user profile
                profile = UserProfile.query.filter_by(user_id=current_user.id).first()
                if profile:
                    # Map model preference to complexity level
                    model_to_complexity = {
                        'simple': 1,
                        'basic': 2,
                        'standard': 3,
                        'advanced': 4,
                        'expert': 5
                    }
                    complexity_level = model_to_complexity.get(profile.preferred_model, 3)
        
        # Process the message using GAKR AI
        # Generate a contextually relevant response based on the user's message
        try:
            # First, try to identify the topic or intent of the message
            if not message_text or len(message_text.strip()) < 3:
                base_response = "Could you please provide more information so I can assist you properly?"
            elif any(keyword in message_text.lower() for keyword in ["hello", "hi", "hey", "greetings"]):
                base_response = "Hello! How can I assist you today with GAKR AI?"
            elif any(keyword in message_text.lower() for keyword in ["thank", "thanks", "appreciate"]):
                base_response = "You're welcome! Is there anything else I can help you with?"
            elif any(keyword in message_text.lower() for keyword in ["bye", "goodbye", "see you"]):
                base_response = "Goodbye! Feel free to return whenever you have more questions."
            elif "?" in message_text:
                # This is likely a question, try to provide a helpful response
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Sample knowledge base (would be more extensive in real implementation)
                knowledge_base = [
                    {"q": "what can you do", "a": "I can answer questions, provide information, and assist with various tasks using natural language processing."},
                    {"q": "how do you work", "a": "I process text using advanced NLP algorithms to understand meaning and generate relevant responses."},
                    {"q": "who made you", "a": "I was created by a team of developers as an AI assistant that runs completely locally without external APIs."},
                    {"q": "what is artificial intelligence", "a": "Artificial Intelligence (AI) refers to systems designed to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and understanding natural language."},
                    {"q": "what is machine learning", "a": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."},
                    {"q": "what is nlp", "a": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language."}
                ]
                
                # Create a simple search function
                questions = [item["q"] for item in knowledge_base]
                answers = [item["a"] for item in knowledge_base]
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer().fit(questions + [message_text])
                question_vectors = vectorizer.transform(questions)
                query_vector = vectorizer.transform([message_text])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, question_vectors)
                
                # Find the most similar question
                most_similar_idx = np.argmax(similarities)
                similarity_score = similarities[0][most_similar_idx]
                
                if similarity_score > 0.3:  # Threshold for relevance
                    base_response = answers[most_similar_idx]
                else:
                    # Generate a generic question response
                    base_response = "That's an interesting question. Based on my analysis, I would say it depends on several factors. Could you provide more context or specify your question further?"
            else:
                # For statements, try to provide a thoughtful response
                if len(message_text.split()) < 5:
                    base_response = "I understand. Could you tell me more about that?"
                else:
                    base_response = "I see what you're saying. Based on my understanding, this relates to concepts in " + \
                                  random.choice(["information processing", "knowledge representation", "decision making", 
                                              "pattern recognition", "language understanding", "cognitive systems"])
        except Exception as e:
            # Fallback in case of any error
            logger.error(f"Error generating response: {str(e)}")
            base_response = "I'm processing your message. Let me think about the best way to respond..."
        
        # Adjust response based on complexity level
        response_text = adjust_response_complexity(base_response, complexity_level)
        
        # Analyze sentiment of user message and response with null check
        if message_text:
            user_sentiment = analyze_text_sentiment(message_text)
        else:
            user_sentiment = {'sentiment': 'neutral', 'compound': 0}
            
        response_sentiment = analyze_text_sentiment(response_text)
        
        # Store the conversation and messages if the user is authenticated
        if current_user.is_authenticated:
            # Get or create conversation
            conversation = None
            if conversation_id:
                # First check if conversation exists and belongs to the current user
                conversation = Conversation.query.get(conversation_id)
                if not conversation or conversation.user_id != current_user.id:
                    # If conversation doesn't exist or doesn't belong to user, create a new one
                    conversation = None
                    flash('Creating a new conversation', 'info')
            
            if not conversation:
                # Create a new conversation
                # Create title with null check
                if message_text:
                    title = message_text[:50] + "..." if len(message_text) > 50 else message_text
                else:
                    title = "New Conversation"
                    
                conversation = Conversation(
                    user_id=current_user.id,
                    title=title
                )
                db.session.add(conversation)
                db.session.commit()
                # Get the newly created conversation ID
                conversation_id = conversation.id
            
            # Add user message with sentiment data
            user_msg = Message(
                conversation_id=conversation.id,
                content=message_text,
                is_user=True
            )
            db.session.add(user_msg)
            
            # Add AI response with sentiment data
            ai_msg = Message(
                conversation_id=conversation.id,
                content=response_text,
                is_user=False
            )
            db.session.add(ai_msg)
            
            # Update conversation timestamp
            conversation.updated_at = datetime.utcnow()
            db.session.commit()
            
            # Auto-train if enabled
            if session.get('auto_train', False):
                # Get existing messages for context
                messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at).all()
                conversation_data = [
                    {'content': msg.content, 'is_user': msg.is_user} 
                    for msg in messages
                ]
                # Start background training
                start_model_training(conversation_data)
            
            # Store sentiment info in session for template rendering
            session['last_user_sentiment'] = {
                'sentiment': user_sentiment['sentiment'],
                'score': user_sentiment['compound'],
                'emoji': user_sentiment.get('sentiment_emoji', '😐')
            }
            
            session['last_response_sentiment'] = {
                'sentiment': response_sentiment['sentiment'],
                'score': response_sentiment['compound'],
                'emoji': response_sentiment.get('sentiment_emoji', '😐')
            }
            
            # Always return to the current conversation ID
            return redirect(url_for('chat.chat', conversation_id=conversation.id))
        
        # For guest users, just return the response without storing
        # Flash the response with sentiment info
        flash_message = f"{response_text}"
        if user_sentiment['sentiment'] != 'neutral':
            flash_message += f" (Your message seemed {user_sentiment['sentiment']})"
            
        flash(flash_message, 'info')
        return redirect(url_for('chat.chat'))
    
    # If form validation fails
    flash('Please enter a valid message.', 'warning')
    return redirect(url_for('chat.chat'))

@chat_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Manage model settings."""
    # Get or create user profile
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.session.add(profile)
        db.session.commit()
    
    # Set form defaults from profile
    form = ModelSettingsForm()
    
    if request.method == 'GET':
        # Set form values from profile
        if profile.preferred_model == 'simple':
            form.complexity_level.data = 1
        elif profile.preferred_model == 'basic':
            form.complexity_level.data = 2
        elif profile.preferred_model == 'advanced':
            form.complexity_level.data = 4
        elif profile.preferred_model == 'expert':
            form.complexity_level.data = 5
        else:
            # Default to standard
            form.complexity_level.data = 3
        
        form.theme.data = profile.theme
        form.show_sentiment.data = profile.spell_check_enabled  # Reusing the field for sentiment display
    
    if form.validate_on_submit():
        # Update profile with form data
        # Map complexity level to model name
        complexity_map = {
            1: 'simple',
            2: 'basic',
            3: 'standard',
            4: 'advanced',
            5: 'expert'
        }
        profile.preferred_model = complexity_map.get(form.complexity_level.data, 'standard')
        profile.theme = form.theme.data
        profile.spell_check_enabled = form.show_sentiment.data
        
        # Store auto-train preference in session
        session['auto_train'] = form.auto_train.data
        
        # Store complexity level in session for easy access
        session['complexity_level'] = form.complexity_level.data
        
        # Save changes
        db.session.commit()
        
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('chat.settings'))
    
    # Get complexity levels for display
    complexity_levels = get_complexity_levels()
    
    return render_template('settings.html', form=form, profile=profile, complexity_levels=complexity_levels)

@chat_bp.route('/clear', methods=['POST'])
@login_required
def clear_history():
    """Clear conversation history."""
    try:
        # Delete all user's conversations
        Conversation.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash('Your conversation history has been cleared.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
    
    return redirect(url_for('chat.history'))

@chat_bp.route('/conversation/<int:conversation_id>/delete', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    """Delete a specific conversation."""
    conversation = Conversation.query.get_or_404(conversation_id)
    
    # Check if the user has access to delete this conversation
    if conversation.user_id != current_user.id:
        abort(403)  # Forbidden
    
    try:
        db.session.delete(conversation)
        db.session.commit()
        flash('Conversation deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
    
    return redirect(url_for('chat.history'))