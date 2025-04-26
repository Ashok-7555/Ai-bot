# GAKR AI Chatbot

GAKR is an AI chatbot that processes and analyzes user text using pre-trained language models with local inference capabilities.

## Features

- Support for multiple AI models without external API dependencies
- Advanced text processing and response generation
- Flexible model loading from various sources
- Sentiment analysis
- Contextual conversation history

## Project Structure

- `main.py`: Main application entry point that launches the AI chatbot web interface
- `core/`: Core functionality including NLP engine and models
- `web_interface/`: Web server implementations (Flask)
- `simple_neural.py`: Simple neural-inspired approach for generating varied responses
- `enhanced_model.py`: Enhanced model interface with improved response generation
- `enhanced_training.py`: Training module for the enhanced model
- `dataset_downloader.py`: Tool to download and prepare datasets for training
- `train_enhanced_model.py`: CLI for training the enhanced model

## Training the Enhanced Model

You can train the enhanced model with various datasets to improve its responses.

### Available Datasets

- **Persona-Chat**: Microsoft Personality Chat dataset with casual and professional responses
- **DailyDialog**: Conversations on common topics
- **Cornell Movie**: Conversations from movie scripts
- Custom training data (`training_data.json`)

### Training Command

```
python train_enhanced_model.py --datasets persona-chat dailydialog cornell-movie --use-existing
```

Options:
- `--datasets`: List of datasets to use for training (default: all available datasets)
- `--use-existing`: Also use existing training data from `training_data.json`
- `--output-dir`: Directory to save the trained model (default: `./models/trained`)
- `--model-name`: Name of the trained model file (default: `simple_model.pkl`)

### Adding Custom Training Data

You can add custom training data by editing the `training_data.json` file or creating a new JSON file with the same format:

```json
[
  {
    "input": "User message here",
    "output": "Bot response here"
  },
  ...
]
```

## Running the Chatbot

```
python main.py
```

This will start the web server on port 5000. You can access the chatbot by opening http://localhost:5000 in your browser.

## Response Generation

The chatbot uses a multi-level response generation approach:

1. **Enhanced Model**: First attempts to use the trained enhanced model
2. **Neural Generator**: Falls back to the simple neural generator if enhanced model fails
3. **Rule-Based**: Uses rule-based approach as final fallback

This ensures robust responses even when trained models are not available.

## Dependencies

- Flask
- Gunicorn
- Other Python standard libraries

## Future Improvements

- Integration with more sophisticated models like GPT-2 when available
- Adding support for more datasets
- Improving the training pipeline
- Enhanced conversation history utilization