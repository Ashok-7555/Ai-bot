from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, RadioField, StringField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from app.models import User, UserProfile
from app.database import db

# Create a blueprint for the authentication routes
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password1 = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password1')])

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class ProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])

class PasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('new_password')])

class PreferencesForm(FlaskForm):
    preferred_model = RadioField('AI Model', choices=[('gpt2', 'Enhanced Model'), ('simple', 'Simple Model')], default='gpt2')
    theme = RadioField('Theme', choices=[('blue', 'Blue'), ('purple', 'Purple'), ('green', 'Green')], default='blue')
    spell_check_enabled = BooleanField('Enable Spell Check', default=True)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        flash('You have been logged in successfully!', 'success')
        
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('main.index')
        return redirect(next_page)
    
    return render_template('login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password1.data)
        
        # Create user profile
        profile = UserProfile(user=user)
        
        db.session.add(user)
        db.session.add(profile)
        db.session.commit()
        
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html', form=form)

@auth_bp.route('/logout')
def logout():
    """Log out the current user."""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/profile', methods=['GET'])
@login_required
def profile():
    """Display user profile."""
    # Get conversation statistics
    conversation_count = len(current_user.conversations)
    message_count = sum([conv.message_count() for conv in current_user.conversations])
    
    # Calculate days active (number of days with conversations)
    active_days = set()
    for conv in current_user.conversations:
        active_days.add(conv.created_at.date())
    days_active = len(active_days)
    
    return render_template('profile.html', 
                          conversation_count=conversation_count,
                          message_count=message_count,
                          days_active=days_active)

@auth_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """User settings page."""
    form = ProfileForm()
    password_form = PasswordForm()
    preferences_form = PreferencesForm()
    
    # Pre-populate the form with the user's current data
    if request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        
        if current_user.profile:
            preferences_form.preferred_model.data = current_user.profile.preferred_model
            preferences_form.theme.data = current_user.profile.theme
            preferences_form.spell_check_enabled.data = current_user.profile.spell_check_enabled
    
    # Handle form submission
    if form.validate_on_submit():
        # Check if username or email has changed
        username_changed = form.username.data != current_user.username
        email_changed = form.email.data != current_user.email
        
        # Check for username uniqueness if changed
        if username_changed:
            existing_user = User.query.filter_by(username=form.username.data).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Username already taken.', 'danger')
                return redirect(url_for('auth.settings'))
        
        # Check for email uniqueness if changed
        if email_changed:
            existing_user = User.query.filter_by(email=form.email.data).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Email already in use.', 'danger')
                return redirect(url_for('auth.settings'))
        
        # Update user data
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('auth.settings'))
    
    return render_template('settings.html', 
                           form=form, 
                           password_form=password_form,
                           preferences_form=preferences_form)

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    form = PasswordForm()
    
    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('auth.settings'))
        
        current_user.set_password(form.new_password.data)
        db.session.commit()
        
        flash('Your password has been updated.', 'success')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{getattr(form, field).label.text}: {error}", 'danger')
    
    return redirect(url_for('auth.settings'))

@auth_bp.route('/update-preferences', methods=['POST'])
@login_required
def update_preferences():
    """Update user preferences."""
    form = PreferencesForm()
    
    if form.validate_on_submit():
        # Ensure the user has a profile
        if not current_user.profile:
            profile = UserProfile(user=current_user)
            db.session.add(profile)
        
        # Update preferences
        current_user.profile.preferred_model = form.preferred_model.data
        current_user.profile.theme = form.theme.data
        current_user.profile.spell_check_enabled = form.spell_check_enabled.data
        current_user.profile.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        flash('Your preferences have been updated.', 'success')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{getattr(form, field).label.text}: {error}", 'danger')
    
    return redirect(url_for('auth.settings'))