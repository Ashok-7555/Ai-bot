"""
GAKR AI - AutoML Manager Module
This module handles AutoML capabilities for model training and optimization.
"""

import os
import json
import logging
import random
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'automl')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class AutoMLManager:
    """
    Manages AutoML capabilities for the GAKR AI chatbot.
    """
    
    def __init__(self):
        """Initialize the AutoML manager."""
        self.training_in_progress = False
        self.training_progress = 0
        self.latest_training_metrics = {}
        self.available_tasks = {
            'sentiment_analysis': {
                'description': 'Determine sentiment of text (positive, negative, neutral)',
                'example_format': {'text': 'string', 'label': 'string'},
                'models': ['LogisticRegression', 'RandomForest', 'SVM', 'NaiveBayes']
            },
            'entity_recognition': {
                'description': 'Identify entities in text (people, locations, organizations)',
                'example_format': {'text': 'string', 'entities': 'list of entity objects'},
                'models': ['CRF', 'BiLSTM-CRF']
            },
            'question_answering': {
                'description': 'Answer questions based on context',
                'example_format': {
                    'context': 'string', 
                    'question': 'string', 
                    'answer': 'string',
                    'answer_start': 'int'
                },
                'models': ['BERT-based', 'RoBERTa-based']
            },
            'conversation_generation': {
                'description': 'Generate responses for conversation',
                'example_format': {'dialogue': 'list of utterances'},
                'models': ['Seq2Seq', 'Transformer-based']
            },
            'image_captioning': {
                'description': 'Generate captions for images',
                'example_format': {'image_id': 'string', 'caption': 'string'},
                'models': ['CNN-LSTM', 'Vision Transformer']
            }
        }
        
        # Track models trained for each task
        self.trained_models = {task: {} for task in self.available_tasks}
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing trained models if available."""
        if os.path.exists(os.path.join(MODELS_DIR, 'model_registry.json')):
            try:
                with open(os.path.join(MODELS_DIR, 'model_registry.json'), 'r') as f:
                    model_info = json.load(f)
                    
                for task, models in model_info.items():
                    for model_name, model_path in models.items():
                        if os.path.exists(model_path):
                            logger.info(f"Found existing model for {task}: {model_name}")
                            # We don't actually load the model until it's needed (lazy loading)
                            self.trained_models[task][model_name] = {
                                'path': model_path,
                                'model': None,
                                'metadata': model_info.get('metadata', {})
                            }
            except Exception as e:
                logger.error(f"Error loading existing models: {str(e)}")
    
    def save_dataset(self, task_name: str, dataset: List[Dict]) -> str:
        """
        Save a dataset for a particular task.
        
        Args:
            task_name: Name of the task
            dataset: List of data examples
            
        Returns:
            Path where the dataset was saved
        """
        if task_name not in self.available_tasks:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(self.available_tasks.keys())}")
        
        # Create timestamp for the dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{task_name}_{timestamp}.json"
        dataset_path = os.path.join(DATA_DIR, dataset_name)
        
        # Save the dataset
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved dataset for {task_name} with {len(dataset)} examples to {dataset_path}")
        return dataset_path
    
    def train_model(self, task_name: str, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a model for a specific task using AutoML techniques.
        
        Args:
            task_name: Name of the task
            dataset: List of data examples
            model_type: Optional model type to use (if None, AutoML will try multiple models)
            
        Returns:
            Dictionary with training results
        """
        if task_name not in self.available_tasks:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(self.available_tasks.keys())}")
        
        if self.training_in_progress:
            return {"status": "error", "message": "Training already in progress"}
        
        self.training_in_progress = True
        self.training_progress = 0
        
        # Save the dataset
        dataset_path = self.save_dataset(task_name, dataset)
        
        # Initialize results
        training_result = {
            "status": "started",
            "task": task_name,
            "dataset_size": len(dataset),
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Select appropriate training method based on task
        try:
            if task_name == 'sentiment_analysis':
                result = self._train_sentiment_analysis(dataset, model_type)
            elif task_name == 'entity_recognition':
                result = self._train_entity_recognition(dataset, model_type)
            elif task_name == 'question_answering':
                result = self._train_question_answering(dataset, model_type)
            elif task_name == 'conversation_generation':
                result = self._train_conversation_generation(dataset, model_type)
            elif task_name == 'image_captioning':
                result = self._train_image_captioning(dataset, model_type)
            else:
                result = {"status": "error", "message": f"Training method not implemented for task: {task_name}"}
            
            # Update training result with model-specific results
            training_result.update(result)
            
            # Update model registry
            self._update_model_registry()
            
        except Exception as e:
            logger.error(f"Error training model for {task_name}: {str(e)}")
            training_result = {
                "status": "error",
                "message": f"Training failed: {str(e)}",
                "task": task_name
            }
        
        self.training_in_progress = False
        self.training_progress = 100
        self.latest_training_metrics = training_result
        
        return training_result
    
    def _train_sentiment_analysis(self, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Train a sentiment analysis model."""
        logger.info(f"Training sentiment analysis model with {len(dataset)} examples")
        
        # Extract features and labels
        texts = [item['text'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Set up models to try
        models = {
            'LogisticRegression': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression(max_iter=1000))
            ]),
            'RandomForest': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ]),
            'SVM': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', SVC(probability=True))
            ]),
            'NaiveBayes': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', MultinomialNB())
            ])
        }
        
        # If model_type is specified, only use that model
        if model_type and model_type in models:
            models_to_try = {model_type: models[model_type]}
        else:
            models_to_try = models
        
        best_model = None
        best_model_name = None
        best_accuracy = 0
        results = {}
        
        # Train each model and evaluate
        for model_name, model in models_to_try.items():
            logger.info(f"Training {model_name} for sentiment analysis")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
        
        # Save the best model
        if best_model:
            model_path = self._save_model(best_model, 'sentiment_analysis', best_model_name)
            
            # Store model reference
            self.trained_models['sentiment_analysis'][best_model_name] = {
                'path': model_path,
                'model': best_model,
                'metadata': {
                    'accuracy': best_accuracy,
                    'training_examples': len(texts),
                    'classes': list(set(labels)),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return {
                "status": "success",
                "best_model": best_model_name,
                "accuracy": best_accuracy,
                "all_results": results,
                "model_path": model_path
            }
        
        return {
            "status": "error",
            "message": "No model could be trained successfully"
        }
    
    def _train_entity_recognition(self, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Train an entity recognition model."""
        logger.info(f"Entity recognition training with {len(dataset)} examples")
        
        # For now, return a simulated result since full NER training is complex
        # In a real implementation, this would use spaCy or similar tools
        return {
            "status": "success",
            "message": "Simulated entity recognition training",
            "accuracy": 0.85,
            "training_examples": len(dataset),
            "entity_types": ["PERSON", "ORG", "GPE", "DATE", "LOC"],
            "model_type": model_type or "CRF"
        }
    
    def _train_question_answering(self, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Train a question answering model."""
        logger.info(f"Question answering training with {len(dataset)} examples")
        
        # For now, return a simulated result since QA training is complex
        # In a real implementation, this would use transformers or similar tools
        return {
            "status": "success",
            "message": "Simulated question answering training",
            "exact_match": 0.76,
            "f1_score": 0.82,
            "training_examples": len(dataset),
            "model_type": model_type or "BERT-based"
        }
    
    def _train_conversation_generation(self, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Train a conversation generation model."""
        logger.info(f"Conversation generation training with {len(dataset)} examples")
        
        # For now, return a simulated result since conversation training is complex
        # In a real implementation, this would use transformer models
        return {
            "status": "success",
            "message": "Simulated conversation generation training",
            "perplexity": 15.2,
            "training_examples": len(dataset),
            "model_type": model_type or "Seq2Seq"
        }
    
    def _train_image_captioning(self, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Train an image captioning model."""
        logger.info(f"Image captioning training with {len(dataset)} examples")
        
        # For now, return a simulated result since image captioning is complex
        # In a real implementation, this would use CNN+LSTM or similar
        return {
            "status": "success",
            "message": "Simulated image captioning training",
            "bleu_score": 0.65,
            "training_examples": len(dataset),
            "model_type": model_type or "CNN-LSTM"
        }
    
    def _save_model(self, model, task_name: str, model_name: str) -> str:
        """Save a trained model to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{task_name}_{model_name}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved {task_name} model ({model_name}) to {model_path}")
        return model_path
    
    def _update_model_registry(self):
        """Update the model registry with current models."""
        registry = {}
        
        for task, models in self.trained_models.items():
            registry[task] = {}
            for model_name, model_info in models.items():
                registry[task][model_name] = model_info['path']
                registry[task][f"{model_name}_metadata"] = model_info.get('metadata', {})
        
        with open(os.path.join(MODELS_DIR, 'model_registry.json'), 'w') as f:
            json.dump(registry, f, indent=2)
    
    def get_model(self, task_name: str, model_name: Optional[str] = None) -> Optional[Any]:
        """
        Get a trained model for a specific task.
        
        Args:
            task_name: Name of the task
            model_name: Optional name of the model (if None, gets the best model)
            
        Returns:
            The model if found, None otherwise
        """
        if task_name not in self.trained_models:
            return None
        
        if not self.trained_models[task_name]:
            return None
        
        # If model_name not specified, get the most recently trained model
        if not model_name:
            model_names = list(self.trained_models[task_name].keys())
            if not model_names:
                return None
            model_name = model_names[-1]
        
        # Check if model exists
        if model_name not in self.trained_models[task_name]:
            return None
        
        model_info = self.trained_models[task_name][model_name]
        
        # Lazy loading - load the model if it hasn't been loaded yet
        if model_info['model'] is None and os.path.exists(model_info['path']):
            try:
                with open(model_info['path'], 'rb') as f:
                    model_info['model'] = pickle.load(f)
                logger.info(f"Loaded {task_name} model ({model_name}) from {model_info['path']}")
            except Exception as e:
                logger.error(f"Error loading model {model_name} for {task_name}: {str(e)}")
                return None
        
        return model_info['model']
    
    def predict(self, task_name: str, input_data: Union[str, Dict], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            task_name: Name of the task
            input_data: Input data for prediction
            model_name: Optional name of the model to use
            
        Returns:
            Dictionary with prediction results
        """
        model = self.get_model(task_name, model_name)
        
        if model is None:
            return {
                "status": "error",
                "message": f"No trained model found for {task_name}"
            }
        
        try:
            if task_name == 'sentiment_analysis':
                # For sentiment analysis, input can be a simple string
                if isinstance(input_data, str):
                    text = input_data
                else:
                    text = input_data.get('text', '')
                
                # Make prediction
                predicted_label = model.predict([text])[0]
                predicted_proba = model.predict_proba([text])[0]
                
                # Get class indices
                classes = model.classes_
                
                # Create probability map
                proba_map = {label: float(prob) for label, prob in zip(classes, predicted_proba)}
                
                return {
                    "status": "success",
                    "task": "sentiment_analysis",
                    "input": text,
                    "prediction": predicted_label,
                    "probabilities": proba_map
                }
            
            # For other tasks, return simulated results for now
            elif task_name == 'entity_recognition':
                return self._simulate_ner_prediction(input_data)
            elif task_name == 'question_answering':
                return self._simulate_qa_prediction(input_data)
            elif task_name == 'conversation_generation':
                return self._simulate_conversation_prediction(input_data)
            elif task_name == 'image_captioning':
                return self._simulate_captioning_prediction(input_data)
            else:
                return {
                    "status": "error",
                    "message": f"Prediction not implemented for task: {task_name}"
                }
                
        except Exception as e:
            logger.error(f"Error in prediction for {task_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    def _simulate_ner_prediction(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """Simulate NER prediction for demonstration."""
        if isinstance(input_data, str):
            text = input_data
        else:
            text = input_data.get('text', '')
        
        # Simplistic entity extraction - not for production use
        entities = []
        if "John" in text or "Mary" in text or "Smith" in text:
            start = text.find("John") if "John" in text else (text.find("Mary") if "Mary" in text else text.find("Smith"))
            end = start + 4 if "John" in text or "Mary" in text else start + 5
            entities.append({"start": start, "end": end, "label": "PERSON", "text": text[start:end]})
        
        if "New York" in text or "London" in text or "Paris" in text:
            for city in ["New York", "London", "Paris"]:
                if city in text:
                    start = text.find(city)
                    end = start + len(city)
                    entities.append({"start": start, "end": end, "label": "GPE", "text": city})
        
        return {
            "status": "success",
            "task": "entity_recognition",
            "input": text,
            "entities": entities
        }
    
    def _simulate_qa_prediction(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """Simulate QA prediction for demonstration."""
        if isinstance(input_data, Dict):
            context = input_data.get('context', '')
            question = input_data.get('question', '')
        else:
            return {
                "status": "error",
                "message": "Question answering requires context and question"
            }
        
        # Very simplistic QA - not for production use
        answer = ""
        answer_start = -1
        
        # For some common questions
        if "capital" in question.lower() and "France" in context:
            answer = "Paris"
            answer_start = context.find(answer)
        elif "capital" in question.lower() and "India" in context:
            answer = "New Delhi"
            answer_start = context.find(answer)
        elif "first" in question.lower() and "president" in question.lower() and "United States" in context:
            answer = "George Washington"
            answer_start = context.find(answer)
        
        return {
            "status": "success",
            "task": "question_answering",
            "context": context,
            "question": question,
            "answer": answer,
            "answer_start": answer_start,
            "confidence": 0.85
        }
    
    def _simulate_conversation_prediction(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """Simulate conversation response generation."""
        if isinstance(input_data, str):
            user_message = input_data
            history = []
        elif isinstance(input_data, Dict):
            user_message = input_data.get('message', '')
            history = input_data.get('history', [])
        else:
            return {
                "status": "error",
                "message": "Invalid input format for conversation generation"
            }
        
        # Simple response generation based on keywords
        responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "how are you": "I'm functioning well, thank you for asking! How about you?",
            "thank": "You're welcome! Is there anything else you need?",
            "bye": "Goodbye! Have a great day!",
            "help": "I'd be happy to help. What do you need assistance with?",
            "weather": "I don't have access to real-time weather data, but I can try to answer other questions.",
            "name": "I'm GAKR AI, your helpful assistant.",
            "joke": "Why don't scientists trust atoms? Because they make up everything!",
            "time": "I don't have access to the current time, but I can help with other information."
        }
        
        response = "I'm not sure how to respond to that. Could you provide more information?"
        
        for keyword, resp in responses.items():
            if keyword in user_message.lower():
                response = resp
                break
        
        return {
            "status": "success",
            "task": "conversation_generation",
            "input": user_message,
            "history": history,
            "response": response
        }
    
    def _simulate_captioning_prediction(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """Simulate image captioning."""
        if isinstance(input_data, Dict):
            image_id = input_data.get('image_id', '')
        else:
            return {
                "status": "error",
                "message": "Image captioning requires an image_id"
            }
        
        # Generic captions
        captions = [
            "A beautiful scene with vibrant colors",
            "An interesting composition of shapes and textures",
            "A detailed view of a natural landscape",
            "A group of people enjoying their time together",
            "A stunning vista with dramatic lighting"
        ]
        
        return {
            "status": "success",
            "task": "image_captioning",
            "image_id": image_id,
            "caption": random.choice(captions),
            "confidence": random.uniform(0.7, 0.95)
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get the current status of model training.
        
        Returns:
            Dictionary with training status
        """
        return {
            "in_progress": self.training_in_progress,
            "progress_percentage": self.training_progress,
            "latest_training": self.latest_training_metrics
        }
    
    def get_available_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available AutoML tasks.
        
        Returns:
            Dictionary with task information
        """
        return self.available_tasks

# Create singleton instance
automl_manager = AutoMLManager()

def get_automl_manager() -> AutoMLManager:
    """Get the AutoML manager instance."""
    return automl_manager

def train_automl_model(task_name: str, dataset: List[Dict], model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Train an AutoML model for a specific task.
    
    Args:
        task_name: Name of the task
        dataset: List of data examples
        model_type: Optional model type to use
        
    Returns:
        Dictionary with training results
    """
    return automl_manager.train_model(task_name, dataset, model_type)

def get_automl_prediction(task_name: str, input_data: Union[str, Dict], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a prediction from an AutoML model.
    
    Args:
        task_name: Name of the task
        input_data: Input data for prediction
        model_name: Optional name of the model to use
        
    Returns:
        Dictionary with prediction results
    """
    return automl_manager.predict(task_name, input_data, model_name)

def get_automl_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available AutoML tasks.
    
    Returns:
        Dictionary with task information
    """
    return automl_manager.get_available_tasks()

def get_automl_training_status() -> Dict[str, Any]:
    """
    Get the current status of AutoML training.
    
    Returns:
        Dictionary with training status
    """
    return automl_manager.get_training_status()