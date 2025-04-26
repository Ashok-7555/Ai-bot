"""
GAKR AI - AI Engine Module
This module contains the AI capabilities of the GAKR chatbot.
"""

__version__ = "1.0.0"

# Import AI engine components
from app.ai_engine.sentiment_analyzer import (
    analyze_text_sentiment,
    analyze_conversation_metrics
)

from app.ai_engine.model_manager import (
    adjust_response_complexity,
    get_complexity_levels,
    start_model_training,
    get_training_status
)

# Import new AutoML components
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