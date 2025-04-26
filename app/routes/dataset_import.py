"""
GAKR AI - Dataset Import Script
Script to import the provided datasets into the system
"""

import json
import logging
from app.ai_engine.dataset_manager import import_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentiment Analysis Dataset
SENTIMENT_DATASET = [
  {"text": "This movie was absolutely fantastic!", "label": "positive"},
  {"text": "I found the service to be quite poor.", "label": "negative"},
  {"text": "The food was okay, nothing special.", "label": "neutral"},
  {"text": "A truly amazing experience from start to finish.", "label": "positive"},
  {"text": "I was very disappointed with the product.", "label": "negative"},
  {"text": "The acting was superb, I really enjoyed it.", "label": "positive"},
  {"text": "This is the worst book I've ever read.", "label": "negative"},
  {"text": "It was an average performance, neither good nor bad.", "label": "neutral"},
  {"text": "Highly recommend this to everyone!", "label": "positive"},
  {"text": "I regret wasting my money on this.", "label": "negative"},
  {"text": "The atmosphere was pleasant and relaxing.", "label": "positive"},
  {"text": "The noise level was unbearable.", "label": "negative"},
  {"text": "It met my expectations, nothing more.", "label": "neutral"},
  {"text": "A delightful and heartwarming story.", "label": "positive"},
  {"text": "The plot was confusing and illogical.", "label": "negative"},
  {"text": "The coffee was lukewarm and weak.", "label": "negative"},
  {"text": "Such a wonderful and inspiring film!", "label": "positive"},
  {"text": "The staff were rude and unhelpful.", "label": "negative"},
  {"text": "It was a decent attempt, but could have been better.", "label": "neutral"},
  {"text": "I absolutely loved every minute of it!", "label": "positive"},
  {"text": "This is simply unacceptable.", "label": "negative"},
  {"text": "The presentation was adequate.", "label": "neutral"},
  {"text": "A truly exceptional piece of work.", "label": "positive"},
  {"text": "I was thoroughly unimpressed.", "label": "negative"},
  {"text": "It was neither good nor bad, just okay.", "label": "neutral"}
]

# Entity Recognition Dataset
ENTITY_DATASET = [
  {"text": "Barack Obama visited India in 2015.", "entities": [{"start": 0, "end": 12, "label": "PERSON"}, {"start": 21, "end": 26, "label": "GPE"}, {"start": 30, "end": 34, "label": "DATE"}]},
  {"text": "Apple is headquartered in Cupertino, California.", "entities": [{"start": 0, "end": 5, "label": "ORG"}, {"start": 25, "end": 34, "label": "GPE"}, {"start": 36, "end": 46, "label": "GPE"}]},
  {"text": "The Eiffel Tower is in Paris, France.", "entities": [{"start": 4, "end": 16, "label": "LOC"}, {"start": 23, "end": 28, "label": "GPE"}, {"start": 30, "end": 36, "label": "GPE"}]},
  {"text": "Sundar Pichai is the CEO of Google.", "entities": [{"start": 0, "end": 13, "label": "PERSON"}, {"start": 25, "end": 31, "label": "ORG"}]},
  {"text": "Amazon was founded by Jeff Bezos in Seattle.", "entities": [{"start": 0, "end": 6, "label": "ORG"}, {"start": 22, "end": 32, "label": "PERSON"}, {"start": 36, "end": 43, "label": "GPE"}]},
  {"text": "The United Nations has its headquarters in New York City.", "entities": [{"start": 4, "end": 18, "label": "ORG"}, {"start": 39, "end": 52, "label": "GPE"}]},
  {"text": "I will be visiting the Taj Mahal next month.", "entities": [{"start": 20, "end": 29, "label": "LOC"}, {"start": 30, "end": 40, "label": "DATE"}]},
  {"text": "Shakespeare wrote Hamlet in the early 17th century.", "entities": [{"start": 0, "end": 11, "label": "PERSON"}, {"start": 18, "end": 24, "label": "WORK_OF_ART"}, {"start": 32, "end": 47, "label": "DATE"}]},
  {"text": "The Ganges River flows through India and Bangladesh.", "entities": [{"start": 4, "end": 17, "label": "LOC"}, {"start": 35, "end": 40, "label": "GPE"}, {"start": 45, "end": 55, "label": "GPE"}]},
  {"text": "Elon Musk is the founder of SpaceX and Tesla.", "entities": [{"start": 0, "end": 9, "label": "PERSON"}, {"start": 28, "end": 34, "label": "ORG"}, {"start": 39, "end": 44, "label": "ORG"}]},
  {"text": "The Louvre Museum is located in Paris.", "entities": [{"start": 4, "end": 17, "label": "ORG"}, {"start": 32, "end": 37, "label": "GPE"}]},
  {"text": "Microsoft released Windows 10 in 2015.", "entities": [{"start": 0, "end": 9, "label": "ORG"}, {"start": 19, "end": 29, "label": "PRODUCT"}, {"start": 33, "end": 37, "label": "DATE"}]},
  {"text": "Mount Everest is the highest peak in the Himalayas.", "entities": [{"start": 0, "end": 12, "label": "LOC"}, {"start": 44, "end": 53, "label": "LOC"}]},
  {"text": "The company Google was founded in California.", "entities": [{"start": 12, "end": 18, "label": "ORG"}, {"start": 34, "end": 44, "label": "GPE"}]},
  {"text": "New Delhi is the capital of India.", "entities": [{"start": 0, "end": 9, "label": "GPE"}, {"start": 25, "end": 30, "label": "GPE"}]},
  {"text": "The iPhone 13 was announced by Apple.", "entities": [{"start": 4, "end": 14, "label": "PRODUCT"}, {"start": 30, "end": 35, "label": "ORG"}]},
  {"text": "Leonardo da Vinci painted the Mona Lisa.", "entities": [{"start": 0, "end": 17, "label": "PERSON"}, {"start": 30, "end": 39, "label": "WORK_OF_ART"}]},
  {"text": "The World Health Organization is based in Geneva.", "entities": [{"start": 4, "end": 30, "label": "ORG"}, {"start": 43, "end": 49, "label": "GPE"}]},
  {"text": "I am planning a trip to Bangalore next week.", "entities": [{"start": 23, "end": 32, "label": "GPE"}, {"start": 33, "end": 42, "label": "DATE"}]},
  {"text": "Sachin Tendulkar is a famous cricketer from India.", "entities": [{"start": 0, "end": 16, "label": "PERSON"}, {"start": 46, "end": 51, "label": "GPE"}]},
  {"text": "The Great Wall of China is a famous landmark.", "entities": [{"start": 4, "end": 25, "label": "LOC"}]},
  {"text": "Toyota is a major car manufacturer from Japan.", "entities": [{"start": 0, "end": 6, "label": "ORG"}, {"start": 41, "end": 46, "label": "GPE"}]}
]

# Question Answering Dataset
QA_DATASET = [
  {
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox jump over?",
    "answer": "the lazy dog",
    "answer_start": 29
  },
  {
    "context": "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
    "question": "Where was Albert Einstein born?",
    "answer": "Ulm, Germany",
    "answer_start": 21
  },
  {
    "context": "The capital of France is Paris.",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "answer_start": 20
  },
  {
    "context": "The first manned flight was achieved by the Wright brothers in 1903.",
    "question": "Who achieved the first manned flight?",
    "answer": "the Wright brothers",
    "answer_start": 37
  },
  {
    "context": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "question": "At what temperature does water boil?",
    "answer": "100 degrees Celsius",
    "answer_start": 15
  },
  {
    "context": "The Indian Space Research Organisation (ISRO) is headquartered in Bangalore.",
    "question": "Where is ISRO headquartered?",
    "answer": "Bangalore",
    "answer_start": 63
  },
  {
    "context": "The currency of Japan is the Yen.",
    "question": "What is the currency of Japan?",
    "answer": "the Yen",
    "answer_start": 24
  },
  {
    "context": "The Amazon rainforest is located in South America.",
    "question": "Where is the Amazon rainforest located?",
    "answer": "South America",
    "answer_start": 35
  },
  {
    "context": "The novel 'Pride and Prejudice' was written by Jane Austen.",
    "question": "Who wrote 'Pride and Prejudice'?",
    "answer": "Jane Austen",
    "answer_start": 42
  },
  {
    "context": "The largest planet in our solar system is Jupiter.",
    "question": "What is the largest planet in our solar system?",
    "answer": "Jupiter",
    "answer_start": 37
  },
  {
    "context": "Photosynthesis is the process by which green plants use sunlight to synthesize foods.",
    "question": "What do green plants use sunlight for in photosynthesis?",
    "answer": "to synthesize foods",
    "answer_start": 56
  },
  {
    "context": "The Taj Mahal was built by Mughal emperor Shah Jahan in memory of his wife.",
    "question": "Who built the Taj Mahal?",
    "answer": "Mughal emperor Shah Jahan",
    "answer_start": 19
  },
  {
    "context": "The Pacific Ocean is the largest ocean on Earth.",
    "question": "Which is the largest ocean on Earth?",
    "answer": "The Pacific Ocean",
    "answer_start": 0
  },
  {
    "context": "Cricket is a popular sport in India, Australia, and England.",
    "question": "In which countries is cricket a popular sport?",
    "answer": "India, Australia, and England",
    "answer_start": 28
  },
  {
    "context": "The theory of relativity was proposed by Albert Einstein.",
    "question": "Who proposed the theory of relativity?",
    "answer": "Albert Einstein",
    "answer_start": 39
  },
  {
    "context": "The first computer programmer was Ada Lovelace.",
    "question": "Who was the first computer programmer?",
    "answer": "Ada Lovelace",
    "answer_start": 29
  },
  {
    "context": "The Himalayas are a mountain range in Asia.",
    "question": "Where are the Himalayas located?",
    "answer": "Asia",
    "answer_start": 37
  },
  {
    "context": "The chemical symbol for water is H2O.",
    "question": "What is the chemical symbol for water?",
    "answer": "H2O",
    "answer_start": 31
  },
  {
    "context": "The Renaissance was a period of great cultural and artistic flourishing in Europe.",
    "question": "Where did the Renaissance flourish?",
    "answer": "Europe",
    "answer_start": 66
  },
  {
    "context": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "question": "What is the approximate speed of light in a vacuum?",
    "answer": "approximately 299,792,458 meters per second",
    "answer_start": 33
  }
]

# Conversation Dataset
CONVERSATION_DATASET = [
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Hello, how are you?"},
      {"speaker": "bot", "utterance": "I am doing well, thank you. How can I help you today?"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What's the weather like in Mumbai?"},
      {"speaker": "bot", "utterance": "The current temperature in Mumbai is 32 degrees Celsius and it is mostly sunny."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Can you tell me a joke?"},
      {"speaker": "bot", "utterance": "Why don't scientists trust atoms? Because they make up everything!"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What time is it?"},
      {"speaker": "bot", "utterance": "The current time is 9:44 PM."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Thank you for your help."},
      {"speaker": "bot", "utterance": "You're welcome!"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "I want to book a flight to Delhi."},
      {"speaker": "bot", "utterance": "Okay, where are you flying from?"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What is the capital of India?"},
      {"speaker": "bot", "utterance": "The capital of India is New Delhi."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Tell me something interesting."},
      {"speaker": "bot", "utterance": "Honey never spoils."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "How do I make a cup of tea?"},
      {"speaker": "bot", "utterance": "First, boil some water. Then, add a tea bag or loose tea leaves and let it steep for a few minutes. You can add milk and sugar if you like."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What are the latest news headlines?"},
      {"speaker": "bot", "utterance": "I'm sorry, I don't have access to real-time news updates."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Could you recommend a good restaurant?"},
      {"speaker": "bot", "utterance": "What kind of cuisine are you in the mood for?"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What is the meaning of life?"},
      {"speaker": "bot", "utterance": "That's a philosophical question with many different answers!"}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Set a reminder for tomorrow at 9 AM."},
      {"speaker": "bot", "utterance": "Okay, I've set a reminder for tomorrow at 9 AM."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What is the capital of Australia?"},
      {"speaker": "bot", "utterance": "The capital of Australia is Canberra."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Play some music."},
      {"speaker": "bot", "utterance": "Okay, playing a random selection now."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "How to learn a new language?"},
      {"speaker": "bot", "utterance": "There are many ways! Consistency, immersion, practice, and using resources like apps and books are all helpful."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What is the weather forecast for Bangalore tomorrow?"},
      {"speaker": "bot", "utterance": "Tomorrow in Bangalore, the forecast is for partly cloudy skies with a high of 34 degrees Celsius and a low of 22 degrees Celsius."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Tell me a fun fact."},
      {"speaker": "bot", "utterance": "Octopuses have three hearts."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Navigate me to the nearest coffee shop."},
      {"speaker": "bot", "utterance": "Sure, I'm finding the nearest coffee shops for you now. One moment."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "What are the symptoms of the common cold?"},
      {"speaker": "bot", "utterance": "Common symptoms include a runny nose, sore throat, cough, and sneezing."}
    ]
  },
  {
    "dialogue": [
      {"speaker": "user", "utterance": "Can you translate 'thank you' to Hindi?"},
      {"speaker": "bot", "utterance": "'Thank you' in Hindi is 'dhanyavaad' (धन्यवाद)."}
    ]
  }
]

# Image Captioning Dataset
CAPTIONING_DATASET = [
  {"image_id": "12345.jpg", "caption": "A fluffy white dog running in a park."},
  {"image_id": "67890.png", "caption": "A colorful sunset over the ocean."},
  {"image_id": "13579.jpeg", "caption": "A plate of delicious pasta with tomatoes and basil."},
  {"image_id": "24680.gif", "caption": "A group of people laughing together."},
  {"image_id": "98765.bmp", "caption": "A majestic mountain range covered in snow."},
  {"image_id": "11223.tiff", "caption": "A close-up of a vibrant red flower."},
  {"image_id": "44556.webp", "caption": "A busy street in a modern city at night."},
  {"image_id": "77889.heic", "caption": "A wooden boat sailing on a calm lake."},
  {"image_id": "10101.raw", "caption": "A baby sleeping peacefully in a crib."},
  {"image_id": "20202.svg", "caption": "A minimalist abstract painting with blue and yellow shapes."},
  {"image_id": "30303.ico", "caption": "A simple icon of a house."},
  {"image_id": "40404.psd", "caption": "A layered image of a landscape with text overlays."},
  {"image_id": "50505.ai", "caption": "A vector illustration of a cartoon character."},
  {"image_id": "60606.indd", "caption": "A page layout design for a magazine article."},
  {"image_id": "70707.xcf", "caption": "An image being edited with various tools visible."},
  {"image_id": "80808.kra", "caption": "A digital painting of a fantasy creature."},
  {"image_id": "90909.blend", "caption": "A 3D rendering of a futuristic vehicle."},
  {"image_id": "11112.obj", "caption": "A wireframe model of a human head."},
  {"image_id": "22223.mtl", "caption": "Material properties for a 3D object."},
  {"image_id": "33334.fbx", "caption": "An animated 3D character walking."},
  {"image_id": "44445.usd", "caption": "A description of a virtual scene with multiple objects."},
  {"image_id": "55556.glb", "caption": "A 3D model of a building that can be viewed in AR."},
  {"image_id": "66667.hdr", "caption": "A high dynamic range image of an interior scene."},
  {"image_id": "77778.exr", "caption": "A multi-layer openEXR image for compositing."},
  {"image_id": "88889.jxr", "caption": "A high-resolution photograph of a bird in flight."},
  {"image_id": "99990.wdp", "caption": "A compressed image with good quality."},
  {"image_id": "12121.jp2", "caption": "An image compressed using JPEG 2000 standard."},
  {"image_id": "23232.tiff", "caption": "A high-fidelity scanned document."},
  {"image_id": "34343.png", "caption": "A transparent image of a logo."},
  {"image_id": "45454.gif", "caption": "A short animated sequence of a bouncing ball."}
]

def import_datasets():
    """Import all datasets into the system."""
    logger.info("Importing datasets...")
    
    # Import sentiment analysis dataset
    logger.info("Importing sentiment analysis dataset...")
    sentiment_result = import_dataset('sentiment_analysis', SENTIMENT_DATASET)
    logger.info(f"Sentiment analysis dataset import result: {sentiment_result}")
    
    # Import entity recognition dataset
    logger.info("Importing entity recognition dataset...")
    entity_result = import_dataset('entity_recognition', ENTITY_DATASET)
    logger.info(f"Entity recognition dataset import result: {entity_result}")
    
    # Import question answering dataset
    logger.info("Importing question answering dataset...")
    qa_result = import_dataset('question_answering', QA_DATASET)
    logger.info(f"Question answering dataset import result: {qa_result}")
    
    # Import conversation dataset
    logger.info("Importing conversation dataset...")
    conversation_result = import_dataset('conversation_generation', CONVERSATION_DATASET)
    logger.info(f"Conversation dataset import result: {conversation_result}")
    
    # Import image captioning dataset
    logger.info("Importing image captioning dataset...")
    captioning_result = import_dataset('image_captioning', CAPTIONING_DATASET)
    logger.info(f"Image captioning dataset import result: {captioning_result}")
    
    logger.info("All datasets imported successfully!")
    
    return {
        'sentiment_analysis': sentiment_result,
        'entity_recognition': entity_result,
        'question_answering': qa_result,
        'conversation_generation': conversation_result,
        'image_captioning': captioning_result
    }

if __name__ == "__main__":
    import_datasets()