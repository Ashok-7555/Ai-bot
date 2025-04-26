import logging
from flask import Blueprint, jsonify, request, session
from flask_login import current_user, login_required

from app.models import UserProfile
from app.database import db

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
settings_bp = Blueprint('settings', __name__, url_prefix='/chat/settings')

@settings_bp.route('/update', methods=['POST'])
@login_required
def update_settings():
    """Update settings via AJAX."""
    try:
        data = request.get_json()
        
        if 'complexity_level' in data:
            complexity_level = int(data['complexity_level'])
            if 1 <= complexity_level <= 5:
                # Update session
                session['complexity_level'] = complexity_level
                
                # Update user profile
                profile = UserProfile.query.filter_by(user_id=current_user.id).first()
                if profile:
                    # Map complexity level to model preference
                    complexity_to_model = {
                        1: 'simple',
                        2: 'basic',
                        3: 'standard',
                        4: 'advanced',
                        5: 'expert'
                    }
                    profile.preferred_model = complexity_to_model.get(complexity_level, 'standard')
                    db.session.commit()
                
                return jsonify({'status': 'success', 'message': 'Complexity level updated'})
        
        if 'theme' in data:
            theme = data['theme']
            valid_themes = ['light', 'dark', 'blue', 'green', 'purple']
            if theme in valid_themes:
                # Update session
                session['theme'] = theme
                
                # Update user profile
                profile = UserProfile.query.filter_by(user_id=current_user.id).first()
                if profile:
                    profile.theme = theme
                    db.session.commit()
                
                return jsonify({'status': 'success', 'message': 'Theme updated'})
        
        if 'auto_train' in data:
            auto_train = bool(data['auto_train'])
            # Update session
            session['auto_train'] = auto_train
            
            return jsonify({'status': 'success', 'message': 'Auto-train setting updated'})
        
        return jsonify({'status': 'error', 'message': 'No valid setting provided'})
    
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})