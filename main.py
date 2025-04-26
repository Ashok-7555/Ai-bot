import os

from flask import Flask, redirect, url_for
from flask_wtf.csrf import CSRFProtect

from app.database import db, login_manager

# Initialize CSRFProtect
csrf = CSRFProtect()

def create_app():
    # Create and configure the Flask application
    app = Flask(__name__)
    
    # Configure the app
    app.config["SECRET_KEY"] = os.environ.get("SESSION_SECRET", "dev-key-for-development-only")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    # Initialize extensions with the app
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    
    # Configure login manager
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.new_chat import chat_bp
    from app.routes.chat_api import chat_api
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(chat_api)
    
    # Create database tables if they don't exist
    with app.app_context():
        from app.models import User, UserProfile, Conversation, Message
        db.create_all()
    
    return app

# Load the user from the database
@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))

# Create the application instance
app = create_app()

# Define a route for the root URL
@app.route('/')
def index():
    return redirect(url_for('main.index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)