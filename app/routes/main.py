from flask import Blueprint, render_template

# Create a blueprint for the main routes
main_bp = Blueprint('main', __name__, url_prefix='/')

@main_bp.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@main_bp.route('/help')
def help_page():
    """Help and documentation page."""
    return render_template('help.html')