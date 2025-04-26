import json
import logging
from typing import Dict, Any, List, Optional

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings

from web_interface.django_app.forms import LoginForm, RegisterForm, SettingsForm
from web_interface.django_app.models import Conversation, Message, UserProfile
from core.nlp_engine import NLPEngine

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize NLP engine
nlp_engine = NLPEngine(model_name=settings.GAKR.get('DEFAULT_MODEL', 'gpt2'))

def index(request: HttpRequest) -> HttpResponse:
    """Render the home page"""
    return render(request, 'index.html')

def chat(request: HttpRequest) -> HttpResponse:
    """Render the chat interface"""
    # Check if user is authenticated
    is_guest = not request.user.is_authenticated
    guest_warning = False
    
    # Get or create conversation for session
    if not is_guest:
        # Get the latest conversation or create a new one for the user
        conversation = Conversation.objects.filter(user=request.user).order_by('-created_at').first()
        if not conversation:
            conversation = Conversation.objects.create(
                user=request.user,
                title="New Conversation"
            )
        
        # Get recent messages for this conversation
        messages = Message.objects.filter(conversation=conversation).order_by('created_at')
        conversation_id = conversation.id
    else:
        # For guests, store conversation in session
        if 'guest_conversation' not in request.session:
            request.session['guest_conversation'] = []
            request.session['guest_message_count'] = 0
        
        messages = request.session['guest_conversation']
        conversation_id = None
        
        # Check if guest is near the conversation limit
        if request.session.get('guest_message_count', 0) >= settings.GAKR.get('GUEST_CONVERSATION_LIMIT', 5):
            guest_warning = True
    
    context = {
        'is_guest': is_guest,
        'messages': messages,
        'conversation_id': conversation_id,
        'guest_warning': guest_warning,
    }
    
    return render(request, 'chat.html', context)

@csrf_exempt
@require_POST
def chat_api(request: HttpRequest) -> JsonResponse:
    """API endpoint for chat interactions"""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not user_message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Process message based on authentication status
        if request.user.is_authenticated:
            # Get the conversation
            if conversation_id:
                conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
            else:
                # Create a new conversation if none exists
                conversation = Conversation.objects.create(
                    user=request.user,
                    title=user_message[:50]  # Use first 50 chars of message as title
                )
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                is_user=True
            )
            
            # Get conversation history for context
            history = []
            previous_messages = Message.objects.filter(conversation=conversation).order_by('created_at')
            for msg in previous_messages:
                if msg.is_user:
                    history.append({'user': msg.content})
                else:
                    history.append({'assistant': msg.content})
            
            # Generate response
            response_text, confidence = nlp_engine.generate_response(user_message, history)
            
            # Save assistant response
            assistant_message = Message.objects.create(
                conversation=conversation,
                content=response_text,
                is_user=False
            )
            
            return JsonResponse({
                'response': response_text,
                'message_id': assistant_message.id,
                'conversation_id': conversation.id,
                'confidence': confidence
            })
            
        else:
            # Guest user flow
            # Update guest message count
            if 'guest_message_count' not in request.session:
                request.session['guest_message_count'] = 0
            request.session['guest_message_count'] += 1
            
            # Check if guest has reached conversation limit
            if request.session['guest_message_count'] > settings.GAKR.get('GUEST_CONVERSATION_LIMIT', 5):
                return JsonResponse({
                    'response': "You've reached the guest message limit. Please register or log in to continue.",
                    'require_login': True
                })
            
            # Get conversation history from session
            if 'guest_conversation' not in request.session:
                request.session['guest_conversation'] = []
            
            history = request.session['guest_conversation']
            
            # Add user message to history
            history.append({'user': user_message, 'timestamp': 'now'})
            
            # Format history for model
            model_history = []
            for msg in history:
                if 'user' in msg:
                    model_history.append({'user': msg['user']})
                if 'assistant' in msg:
                    model_history.append({'assistant': msg['assistant']})
            
            # Generate response
            response_text, confidence = nlp_engine.generate_response(user_message, model_history)
            
            # Add response to history
            history.append({'assistant': response_text, 'timestamp': 'now'})
            
            # Update session
            request.session['guest_conversation'] = history
            request.session.modified = True
            
            return JsonResponse({
                'response': response_text,
                'guest_count': request.session['guest_message_count'],
                'confidence': confidence
            })
            
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def login_view(request: HttpRequest) -> HttpResponse:
    """Handle user login"""
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                
                # Transfer guest conversation if exists
                if 'guest_conversation' in request.session:
                    guest_messages = request.session['guest_conversation']
                    if guest_messages:
                        # Create new conversation
                        conversation = Conversation.objects.create(
                            user=user,
                            title="Imported Guest Conversation"
                        )
                        
                        # Import messages
                        for msg in guest_messages:
                            if 'user' in msg:
                                Message.objects.create(
                                    conversation=conversation,
                                    content=msg['user'],
                                    is_user=True
                                )
                            if 'assistant' in msg:
                                Message.objects.create(
                                    conversation=conversation,
                                    content=msg['assistant'],
                                    is_user=False
                                )
                        
                        # Clear guest session
                        del request.session['guest_conversation']
                        del request.session['guest_message_count']
                
                return redirect('chat')
            else:
                form.add_error(None, "Invalid username or password")
    else:
        form = LoginForm()
    
    return render(request, 'login.html', {'form': form})

def register(request: HttpRequest) -> HttpResponse:
    """Handle user registration"""
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Create user profile
            UserProfile.objects.create(user=user)
            
            # Log in the user
            login(request, user)
            
            # Transfer guest conversation if exists (same as in login)
            if 'guest_conversation' in request.session:
                guest_messages = request.session['guest_conversation']
                if guest_messages:
                    # Create new conversation
                    conversation = Conversation.objects.create(
                        user=user,
                        title="Imported Guest Conversation"
                    )
                    
                    # Import messages
                    for msg in guest_messages:
                        if 'user' in msg:
                            Message.objects.create(
                                conversation=conversation,
                                content=msg['user'],
                                is_user=True
                            )
                        if 'assistant' in msg:
                            Message.objects.create(
                                conversation=conversation,
                                content=msg['assistant'],
                                is_user=False
                            )
                    
                    # Clear guest session
                    del request.session['guest_conversation']
                    del request.session['guest_message_count']
            
            return redirect('chat')
    else:
        form = RegisterForm()
    
    return render(request, 'register.html', {'form': form})

@login_required
def profile(request: HttpRequest) -> HttpResponse:
    """Display and manage user profile"""
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Get conversation statistics
    conversation_count = Conversation.objects.filter(user=request.user).count()
    message_count = Message.objects.filter(conversation__user=request.user).count()
    
    context = {
        'user': request.user,
        'profile': user_profile,
        'conversation_count': conversation_count,
        'message_count': message_count
    }
    
    return render(request, 'profile.html', context)

@login_required
def settings(request: HttpRequest) -> HttpResponse:
    """User settings page"""
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = SettingsForm(request.POST, instance=user_profile)
        if form.is_valid():
            form.save()
            # If the user changed the model, update the NLP engine
            if 'preferred_model' in form.changed_data:
                # This would be handled in a real app, but we'll skip for simplicity
                pass
            return redirect('profile')
    else:
        form = SettingsForm(instance=user_profile)
    
    return render(request, 'settings.html', {'form': form})

@login_required
def history(request: HttpRequest) -> HttpResponse:
    """Display conversation history"""
    conversations = Conversation.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'conversations': conversations})

@login_required
def conversation_detail(request: HttpRequest, conversation_id: int) -> HttpResponse:
    """Display a specific conversation"""
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    messages = Message.objects.filter(conversation=conversation).order_by('created_at')
    
    return render(request, 'chat.html', {
        'conversation': conversation,
        'messages': messages,
        'conversation_id': conversation.id,
        'is_guest': False
    })

@login_required
@require_POST
@csrf_exempt
def clear_history(request: HttpRequest) -> JsonResponse:
    """Clear conversation history"""
    try:
        conversation_id = json.loads(request.body).get('conversation_id')
        
        if conversation_id:
            # Clear specific conversation
            conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
            Message.objects.filter(conversation=conversation).delete()
            return JsonResponse({'success': f'Conversation {conversation_id} cleared'})
        else:
            # Clear all conversations
            Conversation.objects.filter(user=request.user).delete()
            return JsonResponse({'success': 'All conversations cleared'})
            
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required
@csrf_exempt
def history_api(request: HttpRequest) -> JsonResponse:
    """API endpoint for conversation history"""
    if request.method == 'GET':
        conversations = Conversation.objects.filter(user=request.user).order_by('-created_at')
        data = []
        
        for conv in conversations:
            messages = Message.objects.filter(conversation=conv).order_by('created_at')
            data.append({
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat(),
                'message_count': len(messages),
                'preview': messages[0].content[:50] + '...' if messages else ''
            })
            
        return JsonResponse({'conversations': data})
    
    return JsonResponse({'error': 'Invalid method'}, status=405)

def help_page(request: HttpRequest) -> HttpResponse:
    """Help and documentation page"""
    return render(request, 'help.html')
