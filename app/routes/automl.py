"""
GAKR AI - AutoML Routes
Routes for AutoML functionality and dataset management
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Optional as OptionalValidator

from app.ai_engine.automl_manager import (
    get_automl_manager,
    train_automl_model,
    get_automl_prediction,
    get_automl_tasks,
    get_automl_training_status
)

from app.ai_engine.dataset_manager import (
    get_dataset_manager,
    save_dataset,
    load_dataset,
    get_datasets,
    import_dataset
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a blueprint for AutoML routes
automl_bp = Blueprint('automl', __name__, url_prefix='/automl')

# Define the dataset upload form
class DatasetUploadForm(FlaskForm):
    """Form for uploading a dataset."""
    task_type = SelectField('Task Type', validators=[DataRequired()], 
                           choices=[
                               ('sentiment_analysis', 'Sentiment Analysis'),
                               ('entity_recognition', 'Entity Recognition'),
                               ('question_answering', 'Question Answering'),
                               ('conversation_generation', 'Conversation Generation'),
                               ('image_captioning', 'Image Captioning')
                           ])
    
    dataset_name = StringField('Dataset Name', validators=[OptionalValidator()])
    
    dataset_json = TextAreaField('Dataset JSON', validators=[DataRequired()],
                                render_kw={"rows": 15, "placeholder": "[{\"text\": \"Example text\", \"label\": \"positive\"}, ...]"})
    
    submit = SubmitField('Upload Dataset')

# Define the model training form
class ModelTrainingForm(FlaskForm):
    """Form for training a model."""
    task_type = SelectField('Task Type', validators=[DataRequired()])
    
    dataset_name = SelectField('Dataset', validators=[DataRequired()])
    
    model_type = SelectField('Model Type', validators=[OptionalValidator()])
    
    auto_select = BooleanField('Auto-select Best Model', default=True)
    
    submit = SubmitField('Train Model')

@automl_bp.route('/', methods=['GET'])
@login_required
def index():
    """AutoML dashboard."""
    # Get available tasks
    tasks = get_automl_tasks()
    
    # Get available datasets
    datasets = get_datasets()
    
    # Get training status
    training_status = get_automl_training_status()
    
    return render_template(
        'automl/index.html',
        tasks=tasks,
        datasets=datasets,
        training_status=training_status
    )

@automl_bp.route('/datasets', methods=['GET', 'POST'])
@login_required
def datasets():
    """Manage datasets."""
    form = DatasetUploadForm()
    
    if form.validate_on_submit():
        try:
            # Parse the JSON data
            dataset_json = json.loads(form.dataset_json.data)
            
            # Get the task type
            task_type = form.task_type.data
            
            # Get the dataset name (or generate one)
            dataset_name = form.dataset_name.data or None
            
            # Import the dataset
            result = import_dataset(task_type, dataset_json)
            
            if result['status'] == 'success':
                flash(f"Dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
                return redirect(url_for('automl.datasets'))
            else:
                flash(f"Error importing dataset: {result['message']}", 'danger')
                
        except json.JSONDecodeError:
            flash("Invalid JSON format. Please check your data.", 'danger')
        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
    
    # Get available datasets
    datasets = get_datasets()
    
    # Import datasets from the file
    from app.routes.dataset_import import (
        SENTIMENT_DATASET,
        ENTITY_DATASET,
        QA_DATASET,
        CONVERSATION_DATASET,
        CAPTIONING_DATASET
    )
    
    return render_template(
        'automl/datasets.html',
        form=form,
        datasets=datasets,
        sentiment_dataset=SENTIMENT_DATASET,
        entity_dataset=ENTITY_DATASET,
        qa_dataset=QA_DATASET,
        conversation_dataset=CONVERSATION_DATASET,
        captioning_dataset=CAPTIONING_DATASET
    )

@automl_bp.route('/training', methods=['GET', 'POST'])
@login_required
def training():
    """Train models."""
    form = ModelTrainingForm()
    
    # Get available tasks
    tasks = get_automl_tasks()
    
    # Populate task choices
    form.task_type.choices = [(task, tasks[task]['description']) for task in tasks]
    
    # Get available datasets for the selected task
    datasets = get_datasets()
    
    # If task is specified, update dataset choices
    if request.method == 'GET' and request.args.get('task'):
        task = request.args.get('task')
        form.task_type.data = task
        
        # Get datasets for this task
        task_datasets = datasets.get(task, {})
        form.dataset_name.choices = [(name, f"{name} ({info['examples']} examples)") 
                                    for name, info in task_datasets.items()]
        
        # Get model choices for this task
        model_choices = [(model, model) for model in tasks[task]['models']]
        form.model_type.choices = [('', 'Auto-select Best Model')] + model_choices
    
    if form.validate_on_submit():
        try:
            task_type = form.task_type.data
            dataset_name = form.dataset_name.data
            model_type = form.model_type.data if not form.auto_select.data else None
            
            # Load the dataset
            dataset = load_dataset(task_type, dataset_name)
            
            if not dataset:
                flash(f"Dataset not found: {dataset_name}", 'danger')
                return redirect(url_for('automl.training'))
            
            # Start training
            result = train_automl_model(task_type, dataset, model_type)
            
            if result['status'] == 'success':
                flash(f"Model training completed successfully. Best model: {result['best_model']} with accuracy: {result['accuracy']:.2f}", 'success')
            else:
                flash(f"Error training model: {result['message']}", 'danger')
                
            return redirect(url_for('automl.index'))
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
    
    # Get training status
    training_status = get_automl_training_status()
    
    return render_template(
        'automl/training.html',
        form=form,
        tasks=tasks,
        datasets=datasets,
        training_status=training_status
    )

@automl_bp.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    """Make predictions."""
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            task_type = data.get('task_type')
            input_data = data.get('input')
            model_name = data.get('model_name')
            
            if not task_type or not input_data:
                return jsonify({
                    'status': 'error',
                    'message': 'Task type and input data are required'
                }), 400
            
            # Make prediction
            result = get_automl_prediction(task_type, input_data, model_name)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    # Get available tasks
    tasks = get_automl_tasks()
    
    # Get trained models
    automl_manager = get_automl_manager()
    trained_models = automl_manager.trained_models
    
    return render_template(
        'automl/prediction.html',
        tasks=tasks,
        trained_models=trained_models
    )

@automl_bp.route('/api/tasks', methods=['GET'])
@login_required
def api_tasks():
    """Get available tasks."""
    tasks = get_automl_tasks()
    return jsonify(tasks)

@automl_bp.route('/api/datasets', methods=['GET'])
@login_required
def api_datasets():
    """Get available datasets."""
    task = request.args.get('task')
    datasets = get_datasets(task)
    return jsonify(datasets)

@automl_bp.route('/api/training-status', methods=['GET'])
@login_required
def api_training_status():
    """Get training status."""
    status = get_automl_training_status()
    return jsonify(status)

@automl_bp.route('/api/prediction', methods=['POST'])
@login_required
def api_prediction():
    """Make a prediction."""
    try:
        data = request.get_json()
        
        task_type = data.get('task_type')
        input_data = data.get('input')
        model_name = data.get('model_name')
        
        if not task_type or not input_data:
            return jsonify({
                'status': 'error',
                'message': 'Task type and input data are required'
            }), 400
        
        # Make prediction
        result = get_automl_prediction(task_type, input_data, model_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@automl_bp.route('/api/import-dataset', methods=['POST'])
@login_required
def api_import_dataset():
    """Import a dataset."""
    try:
        data = request.get_json()
        
        task_type = data.get('task_type')
        dataset = data.get('dataset')
        dataset_name = data.get('dataset_name')
        
        if not task_type or not dataset:
            return jsonify({
                'status': 'error',
                'message': 'Task type and dataset are required'
            }), 400
        
        # Import dataset
        result = None
        if dataset_name:
            result = save_dataset(task_type, dataset, dataset_name)
        else:
            result = import_dataset(task_type, dataset)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@automl_bp.route('/api/train-model', methods=['POST'])
@login_required
def api_train_model():
    """Train a model."""
    try:
        data = request.get_json()
        
        task_type = data.get('task_type')
        dataset_name = data.get('dataset_name')
        model_type = data.get('model_type')
        
        if not task_type or not dataset_name:
            return jsonify({
                'status': 'error',
                'message': 'Task type and dataset name are required'
            }), 400
        
        # Load dataset
        dataset = load_dataset(task_type, dataset_name)
        
        if not dataset:
            return jsonify({
                'status': 'error',
                'message': f'Dataset not found: {dataset_name}'
            }), 404
        
        # Train model
        result = train_automl_model(task_type, dataset, model_type)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route to import the sentiment dataset provided by the user
@automl_bp.route('/import-sentiment', methods=['POST'])
@login_required
def import_sentiment():
    """Import the sentiment analysis dataset provided by the user."""
    try:
        # This expects a JSON in the request body
        data = request.get_json()
        
        if not data:
            flash("No data provided", 'danger')
            return redirect(url_for('automl.datasets'))
        
        # Import the dataset
        result = import_dataset('sentiment_analysis', data)
        
        if result['status'] == 'success':
            flash(f"Sentiment dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
        else:
            flash(f"Error importing sentiment dataset: {result['message']}", 'danger')
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error importing sentiment dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route to import the entity recognition dataset provided by the user
@automl_bp.route('/import-entities', methods=['POST'])
@login_required
def import_entities():
    """Import the entity recognition dataset provided by the user."""
    try:
        # This expects a JSON in the request body
        data = request.get_json()
        
        if not data:
            flash("No data provided", 'danger')
            return redirect(url_for('automl.datasets'))
        
        # Import the dataset
        result = import_dataset('entity_recognition', data)
        
        if result['status'] == 'success':
            flash(f"Entity recognition dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
        else:
            flash(f"Error importing entity recognition dataset: {result['message']}", 'danger')
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error importing entity recognition dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route to import the question answering dataset provided by the user
@automl_bp.route('/import-qa', methods=['POST'])
@login_required
def import_qa():
    """Import the question answering dataset provided by the user."""
    try:
        # This expects a JSON in the request body
        data = request.get_json()
        
        if not data:
            flash("No data provided", 'danger')
            return redirect(url_for('automl.datasets'))
        
        # Import the dataset
        result = import_dataset('question_answering', data)
        
        if result['status'] == 'success':
            flash(f"Question answering dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
        else:
            flash(f"Error importing question answering dataset: {result['message']}", 'danger')
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error importing question answering dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route to import the conversation dataset provided by the user
@automl_bp.route('/import-conversation', methods=['POST'])
@login_required
def import_conversation():
    """Import the conversation dataset provided by the user."""
    try:
        # This expects a JSON in the request body
        data = request.get_json()
        
        if not data:
            flash("No data provided", 'danger')
            return redirect(url_for('automl.datasets'))
        
        # Import the dataset
        result = import_dataset('conversation_generation', data)
        
        if result['status'] == 'success':
            flash(f"Conversation dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
        else:
            flash(f"Error importing conversation dataset: {result['message']}", 'danger')
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error importing conversation dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route to import the image captioning dataset provided by the user
@automl_bp.route('/import-captioning', methods=['POST'])
@login_required
def import_captioning():
    """Import the image captioning dataset provided by the user."""
    try:
        # This expects a JSON in the request body
        data = request.get_json()
        
        if not data:
            flash("No data provided", 'danger')
            return redirect(url_for('automl.datasets'))
        
        # Import the dataset
        result = import_dataset('image_captioning', data)
        
        if result['status'] == 'success':
            flash(f"Image captioning dataset imported successfully with {result['valid_examples']} valid examples.", 'success')
        else:
            flash(f"Error importing image captioning dataset: {result['message']}", 'danger')
            
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error importing image captioning dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500