"""
GAKR AI - Dataset Manager Module
This module handles dataset loading, saving, and processing.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class DatasetManager:
    """
    Manages datasets for the GAKR AI chatbot.
    """
    
    def __init__(self):
        """Initialize the dataset manager."""
        self.datasets = self._load_existing_datasets()
    
    def _load_existing_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load information about existing datasets."""
        datasets = {}
        
        if os.path.exists(os.path.join(DATA_DIR, 'dataset_registry.json')):
            try:
                with open(os.path.join(DATA_DIR, 'dataset_registry.json'), 'r') as f:
                    datasets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading dataset registry: {str(e)}")
        
        return datasets
    
    def _update_dataset_registry(self):
        """Update the dataset registry with current datasets."""
        with open(os.path.join(DATA_DIR, 'dataset_registry.json'), 'w') as f:
            json.dump(self.datasets, f, indent=2)
    
    def save_dataset(self, task_name: str, dataset: List[Dict], dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Save a dataset for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'sentiment_analysis')
            dataset: List of data examples
            dataset_name: Optional name for the dataset
            
        Returns:
            Dictionary with information about the saved dataset
        """
        if not dataset_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{task_name}_{timestamp}"
        
        # Save the dataset
        dataset_path = os.path.join(DATA_DIR, f"{dataset_name}.json")
        
        try:
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            # Update dataset registry
            if task_name not in self.datasets:
                self.datasets[task_name] = {}
            
            dataset_info = {
                "path": dataset_path,
                "examples": len(dataset),
                "timestamp": datetime.now().isoformat(),
                "name": dataset_name
            }
            
            self.datasets[task_name][dataset_name] = dataset_info
            self._update_dataset_registry()
            
            logger.info(f"Saved dataset {dataset_name} for {task_name} with {len(dataset)} examples")
            
            return {
                "status": "success",
                "task": task_name,
                "dataset_name": dataset_name,
                "examples": len(dataset),
                "path": dataset_path
            }
            
        except Exception as e:
            logger.error(f"Error saving dataset {dataset_name} for {task_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save dataset: {str(e)}"
            }
    
    def load_dataset(self, task_name: str, dataset_name: str) -> Optional[List[Dict]]:
        """
        Load a dataset for a specific task.
        
        Args:
            task_name: Name of the task
            dataset_name: Name of the dataset
            
        Returns:
            List of data examples or None if dataset not found
        """
        if task_name not in self.datasets or dataset_name not in self.datasets[task_name]:
            return None
        
        dataset_info = self.datasets[task_name][dataset_name]
        dataset_path = dataset_info["path"]
        
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded dataset {dataset_name} for {task_name} with {len(dataset)} examples")
            return dataset
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} for {task_name}: {str(e)}")
            return None
    
    def get_datasets(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available datasets.
        
        Args:
            task_name: Optional task name to filter datasets
            
        Returns:
            Dictionary with dataset information
        """
        if task_name:
            return self.datasets.get(task_name, {})
        
        return self.datasets
    
    def import_sentiment_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Import a sentiment analysis dataset.
        
        Args:
            data: List of examples with 'text' and 'label' fields
            
        Returns:
            Dictionary with import results
        """
        # Validate the dataset format
        valid_examples = []
        invalid_examples = []
        
        for i, example in enumerate(data):
            if 'text' in example and 'label' in example:
                valid_examples.append(example)
            else:
                invalid_examples.append({
                    "index": i,
                    "example": example,
                    "reason": "Missing 'text' or 'label' field"
                })
        
        logger.info(f"Importing sentiment dataset: {len(valid_examples)} valid examples, {len(invalid_examples)} invalid examples")
        
        if not valid_examples:
            return {
                "status": "error",
                "message": "No valid examples found",
                "invalid_examples": invalid_examples
            }
        
        # Save the valid examples
        result = self.save_dataset('sentiment_analysis', valid_examples)
        
        if result["status"] == "success":
            result["valid_examples"] = len(valid_examples)
            result["invalid_examples"] = invalid_examples if invalid_examples else []
        
        return result
    
    def import_entity_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Import an entity recognition dataset.
        
        Args:
            data: List of examples with 'text' and 'entities' fields
            
        Returns:
            Dictionary with import results
        """
        # Validate the dataset format
        valid_examples = []
        invalid_examples = []
        
        for i, example in enumerate(data):
            if 'text' in example and 'entities' in example:
                # Validate entity format
                entities_valid = True
                for entity in example['entities']:
                    if 'start' not in entity or 'end' not in entity or 'label' not in entity:
                        entities_valid = False
                        invalid_examples.append({
                            "index": i,
                            "example": example,
                            "reason": "Invalid entity format"
                        })
                        break
                
                if entities_valid:
                    valid_examples.append(example)
            else:
                invalid_examples.append({
                    "index": i,
                    "example": example,
                    "reason": "Missing 'text' or 'entities' field"
                })
        
        logger.info(f"Importing entity dataset: {len(valid_examples)} valid examples, {len(invalid_examples)} invalid examples")
        
        if not valid_examples:
            return {
                "status": "error",
                "message": "No valid examples found",
                "invalid_examples": invalid_examples
            }
        
        # Save the valid examples
        result = self.save_dataset('entity_recognition', valid_examples)
        
        if result["status"] == "success":
            result["valid_examples"] = len(valid_examples)
            result["invalid_examples"] = invalid_examples if invalid_examples else []
        
        return result
    
    def import_qa_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Import a question answering dataset.
        
        Args:
            data: List of examples with 'context', 'question', 'answer', and 'answer_start' fields
            
        Returns:
            Dictionary with import results
        """
        # Validate the dataset format
        valid_examples = []
        invalid_examples = []
        
        for i, example in enumerate(data):
            if all(field in example for field in ['context', 'question', 'answer', 'answer_start']):
                valid_examples.append(example)
            else:
                invalid_examples.append({
                    "index": i,
                    "example": example,
                    "reason": "Missing required field"
                })
        
        logger.info(f"Importing QA dataset: {len(valid_examples)} valid examples, {len(invalid_examples)} invalid examples")
        
        if not valid_examples:
            return {
                "status": "error",
                "message": "No valid examples found",
                "invalid_examples": invalid_examples
            }
        
        # Save the valid examples
        result = self.save_dataset('question_answering', valid_examples)
        
        if result["status"] == "success":
            result["valid_examples"] = len(valid_examples)
            result["invalid_examples"] = invalid_examples if invalid_examples else []
        
        return result
    
    def import_conversation_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Import a conversation dataset.
        
        Args:
            data: List of examples with 'dialogue' field (list of utterances)
            
        Returns:
            Dictionary with import results
        """
        # Validate the dataset format
        valid_examples = []
        invalid_examples = []
        
        for i, example in enumerate(data):
            if 'dialogue' in example and isinstance(example['dialogue'], list):
                valid_dialogue = True
                for utterance in example['dialogue']:
                    if 'speaker' not in utterance or 'utterance' not in utterance:
                        valid_dialogue = False
                        invalid_examples.append({
                            "index": i,
                            "example": example,
                            "reason": "Invalid dialogue format"
                        })
                        break
                
                if valid_dialogue:
                    valid_examples.append(example)
            else:
                invalid_examples.append({
                    "index": i,
                    "example": example,
                    "reason": "Missing 'dialogue' field or not a list"
                })
        
        logger.info(f"Importing conversation dataset: {len(valid_examples)} valid examples, {len(invalid_examples)} invalid examples")
        
        if not valid_examples:
            return {
                "status": "error",
                "message": "No valid examples found",
                "invalid_examples": invalid_examples
            }
        
        # Save the valid examples
        result = self.save_dataset('conversation_generation', valid_examples)
        
        if result["status"] == "success":
            result["valid_examples"] = len(valid_examples)
            result["invalid_examples"] = invalid_examples if invalid_examples else []
        
        return result
    
    def import_captioning_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Import an image captioning dataset.
        
        Args:
            data: List of examples with 'image_id' and 'caption' fields
            
        Returns:
            Dictionary with import results
        """
        # Validate the dataset format
        valid_examples = []
        invalid_examples = []
        
        for i, example in enumerate(data):
            if 'image_id' in example and 'caption' in example:
                valid_examples.append(example)
            else:
                invalid_examples.append({
                    "index": i,
                    "example": example,
                    "reason": "Missing 'image_id' or 'caption' field"
                })
        
        logger.info(f"Importing captioning dataset: {len(valid_examples)} valid examples, {len(invalid_examples)} invalid examples")
        
        if not valid_examples:
            return {
                "status": "error",
                "message": "No valid examples found",
                "invalid_examples": invalid_examples
            }
        
        # Save the valid examples
        result = self.save_dataset('image_captioning', valid_examples)
        
        if result["status"] == "success":
            result["valid_examples"] = len(valid_examples)
            result["invalid_examples"] = invalid_examples if invalid_examples else []
        
        return result

# Create singleton instance
dataset_manager = DatasetManager()

def get_dataset_manager() -> DatasetManager:
    """Get the dataset manager instance."""
    return dataset_manager

def save_dataset(task_name: str, dataset: List[Dict], dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Save a dataset for a specific task.
    
    Args:
        task_name: Name of the task
        dataset: List of data examples
        dataset_name: Optional name for the dataset
        
    Returns:
        Dictionary with information about the saved dataset
    """
    return dataset_manager.save_dataset(task_name, dataset, dataset_name)

def load_dataset(task_name: str, dataset_name: str) -> Optional[List[Dict]]:
    """
    Load a dataset for a specific task.
    
    Args:
        task_name: Name of the task
        dataset_name: Name of the dataset
        
    Returns:
        List of data examples or None if dataset not found
    """
    return dataset_manager.load_dataset(task_name, dataset_name)

def get_datasets(task_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about available datasets.
    
    Args:
        task_name: Optional task name to filter datasets
        
    Returns:
        Dictionary with dataset information
    """
    return dataset_manager.get_datasets(task_name)

def import_dataset(task_name: str, data: List[Dict]) -> Dict[str, Any]:
    """
    Import a dataset for a specific task.
    
    Args:
        task_name: Name of the task
        data: List of data examples
        
    Returns:
        Dictionary with import results
    """
    if task_name == 'sentiment_analysis':
        return dataset_manager.import_sentiment_dataset(data)
    elif task_name == 'entity_recognition':
        return dataset_manager.import_entity_dataset(data)
    elif task_name == 'question_answering':
        return dataset_manager.import_qa_dataset(data)
    elif task_name == 'conversation_generation':
        return dataset_manager.import_conversation_dataset(data)
    elif task_name == 'image_captioning':
        return dataset_manager.import_captioning_dataset(data)
    else:
        return {
            "status": "error",
            "message": f"Unknown task: {task_name}"
        }