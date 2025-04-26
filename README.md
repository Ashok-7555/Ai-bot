# GAKR AI - Transformer-based Chatbot

GAKR AI is a Django-based AI chatbot application that uses transformer models to process and respond to user prompts without requiring external APIs. The application provides a clean and intuitive interface for users to chat with an AI assistant.

## Features

- **Natural Language Processing**: Utilizes Hugging Face transformers for text generation and understanding
- **User Accounts**: Registration, login, and profile management
- **Conversation History**: Save and retrieve past conversations
- **Responsive Design**: Works on desktop and mobile devices
- **Spell Checking**: Automatic correction of spelling errors
- **Guest Mode**: Try the chatbot without registering
- **Settings Customization**: Change themes, models, and preferences

## Technical Stack

- **Backend**:
  - Python 3.9+
  - Django web framework
  - Transformers library (Hugging Face)
  - PyTorch for model inference
  - ChromaDB for vector storage
  - NLTK and spaCy for text processing

- **Frontend**:
  - HTML/CSS/JavaScript
  - Bootstrap 5
  - FontAwesome icons

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gakr-ai.git
   cd gakr-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python manage.py migrate
   ```

4. Start the server:
   ```bash
   python manage.py runserver 0.0.0.0:5000
   ```

5. Access the application:
   Open your browser and navigate to `http://localhost:5000`

## Usage

### Initial Setup

When you first access GAKR AI, you can try it as a guest or create an account:

1. **Guest Mode**: Click "Try as Guest" on the homepage to start chatting immediately. Guest mode is limited to 5 exchanges.
2. **Register**: Create an account to access all features, including unlimited conversations and history tracking.
3. **Login**: If you already have an account, log in to continue your conversations.

### Interacting with GAKR

- **Asking Questions**: Type your query in the chat input box and press Enter.
- **Conversation Context**: GAKR remembers the context of your conversation, so you can ask follow-up questions.
- **Viewing History**: Access your past conversations via the History page (registered users only).
- **Customizing Settings**: Change preferences via the Settings page (registered users only).

## Models

GAKR AI uses several transformer models:

- **Default**: GPT-2 (small but versatile)
- **Alternatives**:
  - DistilGPT-2 (faster, smaller model)
  - DialoGPT (conversation-focused model)
- **Kaggle Models**: Can use specific models like "qwen-lm/qwq-32b"

Models are loaded and used without requiring external API keys.

## Project Structure

