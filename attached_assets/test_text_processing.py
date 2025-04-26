#!/usr/bin/env python3
"""
Test script for the GAKR AI text processing pipeline.
This script tests the functionality of the core text processing components.
"""

import logging
from core.utils.text_processing import process_input, process_output
from core.model_inference import generate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_text_processing_pipeline():
    """Test the complete text processing pipeline."""
    logger.info("Testing text processing pipeline...")
    
    # Test 1: Basic input processing
    logger.info("Test 1: Basic input processing")
    test_input = "Hello, how are you doing today? I'm interested in learning about AI."
    
    processed_input = process_input(test_input)
    
    logger.info(f"Original input: {test_input}")
    logger.info(f"Processed text: {processed_input['text']}")
    logger.info(f"Tokens: {processed_input['tokens']}")
    logger.info(f"Context: {processed_input['context']}")
    logger.info("=" * 50)
    
    # Test 2: Process with conversation history
    logger.info("Test 2: Process with conversation history")
    conversation_history = [
        {"type": "user", "message": "What can you tell me about machine learning?"},
        {"type": "bot", "message": "Machine learning is a field of AI focused on developing systems that learn from data."},
        {"type": "user", "message": "That sounds interesting!"}
    ]
    
    test_input_2 = "How is it different from deep learning?"
    processed_input_2 = process_input(test_input_2, conversation_history)
    
    logger.info(f"Original input: {test_input_2}")
    logger.info(f"Processed text: {processed_input_2['text']}")
    logger.info(f"Context: {processed_input_2['context']}")
    logger.info("=" * 50)
    
    # Test 3: Output processing
    logger.info("Test 3: Output processing")
    raw_output = "this is a raw output from the model without proper capitalization or punctuation"
    processed_output = process_output(raw_output)
    
    logger.info(f"Raw output: {raw_output}")
    logger.info(f"Processed output: {processed_output}")
    logger.info("=" * 50)
    
    # Test 4: Dictionary output processing
    logger.info("Test 4: Dictionary output processing")
    dict_output = {
        "response": "here's a response in a dictionary format",
        "confidence": 0.85,
        "model": "simple"
    }
    processed_dict_output = process_output(dict_output)
    
    logger.info(f"Raw dictionary output: {dict_output}")
    logger.info(f"Processed output: {processed_dict_output}")
    logger.info("=" * 50)
    
    # Test 5: Full pipeline with generate_response
    logger.info("Test 5: Full pipeline with model inference")
    try:
        full_response = generate_response(
            "Tell me about artificial intelligence", 
            conversation_history=conversation_history
        )
        
        logger.info(f"Full pipeline response: {full_response}")
        
        if isinstance(full_response, dict):
            logger.info(f"Response text: {full_response.get('response', '')}")
            logger.info(f"Model used: {full_response.get('model', 'unknown')}")
            logger.info(f"Generation time: {full_response.get('generation_time', 0)} seconds")
            logger.info(f"Confidence: {full_response.get('confidence', 0)}")
        else:
            logger.info(f"Response (string): {full_response}")
    except Exception as e:
        logger.error(f"Error testing full pipeline: {e}")
    
    logger.info("=" * 50)
    logger.info("Text processing pipeline tests completed")

if __name__ == "__main__":
    test_text_processing_pipeline()