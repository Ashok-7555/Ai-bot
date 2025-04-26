from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    """Extended user profile for GAKR AI"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    preferred_model = models.CharField(max_length=100, default='gpt2')
    theme = models.CharField(max_length=50, default='blue')
    spell_check_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"

class Conversation(models.Model):
    """Conversation model to group messages"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"
    
    def message_count(self):
        """Get count of messages in this conversation"""
        return self.messages.count()

class Message(models.Model):
    """Individual message in a conversation"""
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)  # True if message is from user, False if from AI
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        sender = "User" if self.is_user else "GAKR"
        return f"{sender}: {self.content[:30]}..."
