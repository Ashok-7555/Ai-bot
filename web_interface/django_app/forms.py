from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from web_interface.django_app.models import UserProfile

class LoginForm(forms.Form):
    """User login form"""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )

class RegisterForm(UserCreationForm):
    """User registration form"""
    email = forms.EmailField(
        max_length=254,
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email Address'})
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        
    def __init__(self, *args, **kwargs):
        super(RegisterForm, self).__init__(*args, **kwargs)
        # Add appropriate CSS classes to all fields
        for field_name in self.fields:
            self.fields[field_name].widget.attrs['class'] = 'form-control'
            if field_name == 'username':
                self.fields[field_name].widget.attrs['placeholder'] = 'Username'
            elif field_name == 'password1':
                self.fields[field_name].widget.attrs['placeholder'] = 'Password'
            elif field_name == 'password2':
                self.fields[field_name].widget.attrs['placeholder'] = 'Confirm password'

class SettingsForm(forms.ModelForm):
    """User settings form"""
    THEME_CHOICES = [
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('blue', 'Blue')
    ]
    
    MODEL_CHOICES = [
        ('gpt2', 'Default (GPT-2)'),
        ('distilgpt2', 'DistilGPT-2 (Faster)'),
        ('microsoft/DialoGPT-small', 'DialoGPT (Conversation-focused)')
    ]
    
    theme = forms.ChoiceField(choices=THEME_CHOICES, required=True)
    preferred_model = forms.ChoiceField(choices=MODEL_CHOICES, required=True)
    spell_check_enabled = forms.BooleanField(required=False)
    
    class Meta:
        model = UserProfile
        fields = ['theme', 'preferred_model', 'spell_check_enabled']
        
    def __init__(self, *args, **kwargs):
        super(SettingsForm, self).__init__(*args, **kwargs)
        # Add appropriate CSS classes to all fields
        for field_name in self.fields:
            self.fields[field_name].widget.attrs['class'] = 'form-control'
