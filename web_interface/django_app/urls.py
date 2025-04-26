"""
URL configuration for GAKR AI project.
"""

from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from web_interface.django_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Main pages
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register, name='register'),
    
    # User profile and settings
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings, name='settings'),
    
    # Conversation history
    path('history/', views.history, name='history'),
    path('history/<int:conversation_id>/', views.conversation_detail, name='conversation_detail'),
    
    # Help and documentation
    path('help/', views.help_page, name='help'),
    
    # API endpoints
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/history/', views.history_api, name='history_api'),
    path('api/clear-history/', views.clear_history, name='clear_history'),
]
