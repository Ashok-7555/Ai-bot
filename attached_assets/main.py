"""
GAKR - Main application entry point.
Launches the AI chatbot web interface.
"""

import logging
import os
import sys
import json
# No threading needed since we're not loading models in background
from flask import Flask, render_template, request, jsonify, session, redirect

# Import our enhanced model and perplexity integration
try:
    from enhanced_model import generate_enhanced_response
    enhanced_model_available = True
except ImportError:
    enhanced_model_available = False
    logging.warning("Enhanced model not available")

# Try to import the perplexity helper 
perplexity_helper = None
try:
    import perplexity_helper
    perplexity_initialized = True
except ImportError:
    perplexity_initialized = False
    logging.warning("Perplexity integration not available")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting GAKR AI Chatbot")

# Initialize Flask app
app = Flask(__name__, 
           static_folder='web_interface/static',
           template_folder='web_interface/templates')
app.secret_key = os.environ.get("SESSION_SECRET", "gakr-dev-secret-key")
logger.info("Flask app initialized")

# Helper function to check if user is logged in
def is_authenticated():
    return session.get('logged_in', False)

# Chat history storage (in-memory for now)
chat_histories = {}

# Flag to indicate if the model is loaded
kaggle_model_loaded = False
kaggle_model_path = None
kaggle_model_loading = False

# Dictionary of predefined responses for various topics for fallback
knowledge_base = {
    # Programming Languages
    "java": "Java is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It is a general-purpose programming language intended to let application developers write once, run anywhere, meaning that compiled Java code can run on all platforms that support Java without the need for recompilation.",
    "python": "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.",
    "javascript": "JavaScript is a programming language that enables interactive web pages and is an essential part of web applications. It's a core technology of the World Wide Web and is essential for web applications.",
    "c++": "C++ is a general-purpose programming language created as an extension of the C programming language. It has object-oriented, generic, and functional features in addition to facilities for low-level memory manipulation.",
    "ruby": "Ruby is an interpreted, high-level, general-purpose programming language that supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "php": "PHP is a general-purpose scripting language especially suited to web development. It was originally created by Danish-Canadian programmer Rasmus Lerdorf in 1994.",

    # Web Technologies
    "html": "HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser. It defines the structure and content of web pages.",
    "css": "CSS (Cascading Style Sheets) is a style sheet language used for describing the presentation of a document written in HTML. CSS is designed to enable the separation of content and presentation.",
    "frontend": "Frontend development refers to building the visible parts of a website that users interact with. Key technologies include HTML (structure), CSS (styling), and JavaScript (interactivity). Frontend developers work on user interfaces, responsive designs, and ensuring good user experience across different devices and browsers.",
    "backend": "Backend development refers to the server-side of web development. It involves working with servers, databases, APIs, and application logic that users don't directly interact with. Common backend languages include Python, Java, Ruby, PHP, and Node.js.",
    
    # Computing Concepts
    "computer": "A computer is an electronic device that manipulates information or data. It has the ability to store, retrieve, and process data.",
    "programming": "Programming is the process of creating a set of instructions that tell a computer how to perform a task.",
    "code": "Code refers to the instructions written in a programming language that a computer can execute.",
    "software": "Software refers to programs and other operating information used by a computer.",
    "hardware": "Hardware refers to the physical components of a computer system.",
    "app": "An app (application) is a type of software that allows you to perform specific tasks on your computer or mobile device.",
    "website": "A website is a collection of web pages accessible through the internet, typically served from a single domain name.",
    "internet": "The Internet is the global system of interconnected computer networks that use the Internet protocol suite to link devices worldwide.",
    "cloud computing": "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, analytics, and intelligence—over the Internet ('the cloud') to offer faster innovation, flexible resources, and economies of scale.",
    
    # AI and Data Science
    "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    "machine learning": "Machine Learning is a branch of artificial intelligence that focuses on using data and algorithms to improve accuracy progressively. It involves computer algorithms that can access data and use it to learn for themselves.",
    "deep learning": "Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. It can be supervised, semi-supervised or unsupervised and is particularly useful for processing large amounts of unstructured data.",
    "data science": "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "big data": "Big Data refers to extremely large data sets that may be analyzed computationally to reveal patterns, trends, and associations, especially relating to human behavior and interactions.",
    "neural network": "A Neural Network is a computational model based on the structure and functions of biological neural networks. It's a core component of deep learning algorithms and is used for pattern recognition and feature learning.",
    "natural language processing": "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
    "text mining": "Text Mining is the process of deriving high-quality information from text. It involves using data mining algorithms to find patterns and relationships within textual data. Text mining tasks include text categorization, clustering, concept extraction, sentiment analysis, and document summarization.",
    
    # Engineering Fields and Education
    "computer science": "Computer Science (CS) is the study of computers and computational systems. It involves the theoretical foundations of information and computation, together with practical techniques for their implementation and application.",
    "software engineering": "Software Engineering is the systematic application of engineering approaches to the development of software. It involves applying engineering principles to software creation, including design, development, testing, and maintenance.",
    "data engineering": "Data Engineering is the aspect of data science that focuses on practical applications of data collection and analysis. Data engineers build systems that collect, manage, and convert raw data into usable information for data scientists and business analysts to interpret.",
    "cse": "Computer Science and Engineering (CSE) is an academic program that integrates the field of computer science and computer engineering. The program typically covers both the theoretical and practical aspects of computing.",
    "ece": "Electronics and Communication Engineering (ECE) is a discipline that focuses on the design, development, and maintenance of electronic equipment and communication systems. It covers areas such as telecommunications, radio engineering, and signal processing.",
    "electrical engineering": "Electrical Engineering is a field of engineering that deals with the study and application of electricity, electronics, and electromagnetism. It covers power generation, transmission, and the application of electrical systems.",
    "btech": "Bachelor of Technology (BTech) is an undergraduate academic degree conferred after completion of a four-year program in engineering or technology fields. It focuses on applying technological and scientific knowledge to solve engineering problems.",
    "mtech": "Master of Technology (MTech) is a postgraduate academic degree awarded to candidates after completion of a two-year program in engineering or technology fields. It allows for specialization and advanced study in specific areas of technology.",
    "phd": "PhD (Doctor of Philosophy) is the highest academic degree awarded for original research in a specific field of study. It typically requires several years of study and research, culminating in a dissertation that contributes new knowledge to the field.",
    
    # Business and Commerce
    "e-commerce": "E-commerce (electronic commerce) refers to the buying and selling of goods or services using the internet, and the transfer of money and data to execute these transactions.",
    "digital marketing": "Digital Marketing is the component of marketing that utilizes internet and online based digital technologies such as desktop computers, mobile phones and other digital media and platforms to promote products and services.",
    "business": "Business refers to an organization or enterprising entity engaged in commercial, industrial, or professional activities. Businesses can be for-profit entities or non-profit organizations.",
    "entrepreneurship": "Entrepreneurship is the process of designing, launching, and running a new business, which typically begins as a small business offering a product, process, or service for sale.",
    
    # Other Technology
    "blockchain": "Blockchain is a system of recording information in a way that makes it difficult or impossible to change, hack, or cheat the system. It is a digital ledger of transactions that is duplicated and distributed across the entire network of computer systems.",
    "iot": "Internet of Things (IoT) refers to the network of physical objects—'things'—that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet.",
    "ar": "Augmented Reality (AR) is an interactive experience of a real-world environment where the objects that reside in the real world are enhanced by computer-generated perceptual information.",
    "vr": "Virtual Reality (VR) is a simulated experience that can be similar to or completely different from the real world. It uses computer technology to create a simulated environment.",
    "cybersecurity": "Cybersecurity refers to the practice of protecting systems, networks, and programs from digital attacks. These attacks are usually aimed at accessing, changing, or destroying sensitive information.",
    "cloud": "Cloud computing is a technology that allows users to access and use computing resources (like servers, storage, databases, networking, software) over the internet, instead of owning and maintaining physical infrastructure.",
    
    # Symbols and Characters
    "@": "The at sign (@) is a symbol used in email addresses to separate the username from the domain name. It's also used in social media platforms to tag or mention users (like @username), and in programming languages for various purposes depending on the language.",
    "#": "The hash symbol (#) is used for various purposes including: as a number sign, to indicate a hashtag on social media platforms to categorize content, in programming languages to denote comments or preprocessor directives, and in music notation to represent a sharp note.",
    "&": "The ampersand (&) is a symbol that represents the word 'and'. It's commonly used in business names, programming (as a logical AND operator in some languages), and in HTML to represent special characters (e.g., &amp; represents the & symbol itself).",
    
    # Entertainment and Media
    "movie": "A movie (or film) is a series of still images that, when shown on a screen, create the illusion of moving images. Movies are a popular form of entertainment and art that tells stories or documentaries through the combination of images, sound, and special effects.",
    "music": "Music is an art form consisting of sound organized in time. It's a universal language that can express emotions, tell stories, and bring people together through rhythms, melodies, and harmonies produced by instruments or vocals.",
    "game": "A game is a structured form of play, usually undertaken for entertainment or fun, and sometimes used as an educational tool. Games typically involve rules, challenges, and interaction, and can be played alone, in teams, or online.",
    "book": "A book is a medium for recording information in the form of writing or images, typically composed of many pages bound together and protected by a cover. Books can contain fiction, non-fiction, poetry, reference material, or instructional content.",
    "library": "A library is a collection of sources of information and similar resources, made accessible to a defined community for reference or borrowing. It provides physical or digital access to material, and may be a physical location or a virtual space.",
    "database": "A database is an organized collection of structured information, or data, typically stored electronically in a computer system. Databases are designed to efficiently manage, store, and retrieve data, and are fundamental to many applications and services.",
    
    # Animals and Nature
    "animals": "Animals are multicellular, eukaryotic organisms in the biological kingdom Animalia. They include mammals, birds, reptiles, amphibians, fish, and invertebrates like insects and crustaceans. Animals are characterized by their ability to move, consume organic material, and respond to stimuli.",
    "plants": "Plants are mainly multicellular organisms in the kingdom Plantae that use photosynthesis to make their own food. They include familiar organisms such as trees, flowers, herbs, bushes, grasses, vines, ferns, mosses, and green algae.",
    "environment": "The environment encompasses all living and non-living things occurring naturally on Earth. It includes physical, chemical, and other natural forces, as well as all living things that interact with these forces. The environment is increasingly affected by human activities.",
    
    # Chatbot Related
    "chatbot": "A chatbot is a computer program that simulates and processes human conversation, either written or spoken, allowing humans to interact with digital devices as if they were communicating with a real person.",
    "gakr": "GAKR is an AI chatbot built to process and analyze text using pre-trained models without requiring external API dependencies. It can provide information, answer questions, and engage in conversations about various topics.",
    
    # Math Operations
    "math": {
        "addition": lambda a, b: float(a) + float(b),
        "subtraction": lambda a, b: float(a) - float(b),
        "multiplication": lambda a, b: float(a) * float(b),
        "division": lambda a, b: float(a) / float(b) if float(b) != 0 else "Cannot divide by zero"
    }
}

# Set initial values for Kaggle model flags
kaggle_model_loaded = False
kaggle_model_path = None
kaggle_model_loading = False

logger.info("Kaggle model loading disabled due to disk quota limitations")

@app.route("/")
def index():
    """Render the main page."""
    logger.info("Rendering index page")
    session['guest'] = True
    # Automatically mark onboarding as completed for all users
    session['onboarding_completed'] = True
    return render_template("index.html")

@app.route("/chat")
def chat():
    """Render the chat interface."""
    logger.info("Rendering chat page")
    if session.get('logged_in'):
        return render_template("user_chat.html", 
                             username=session.get('username', 'User'),
                             email=session.get('email', 'user@example.com'))
    else:
        session['guest'] = True
        return render_template("chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Render the auth page or process login."""
    logger.info("Processing login")
    login_error = None
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # In a real application, validate credentials against database
        # For now, simple mock login
        if email and password:
            session['logged_in'] = True
            session['username'] = email.split('@')[0]  # Use part before @ as username
            session['email'] = email
            session['guest'] = False
            return redirect('/chat')
        else:
            login_error = "Invalid email or password"
            
    return render_template("auth.html", login_error=login_error)

@app.route("/register", methods=["GET", "POST"])
def register():
    """Process registration."""
    logger.info("Processing registration")
    register_error = None
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # In a real application, validate and save user to database
        if not name or not email or not password:
            register_error = "All fields are required"
        elif password != confirm_password:
            register_error = "Passwords do not match"
        else:
            # Registration successful
            session['logged_in'] = True
            session['username'] = name
            session['email'] = email
            session['guest'] = False
            return redirect('/chat')
    
    # For GET requests, redirect to login page with signup parameter
    if request.method == 'GET':
        return redirect('/login?signup=true')
    else:
        # For failed POST, show with error
        return render_template("auth.html", register_error=register_error)

@app.route("/profile")
def profile():
    """Render the user profile page."""
    logger.info("Rendering profile page")
    if not session.get('logged_in'):
        return redirect('/login')
        
    return render_template("profile.html", 
                          username=session.get('username', 'User'),
                          email=session.get('email', 'user@example.com'))

@app.route("/logout")
def logout():
    """Log the user out."""
    logger.info("Logging out user")
    session.clear()
    return redirect('/')

@app.route("/onboarding")
def onboarding():
    """Show the onboarding wizard."""
    logger.info("Showing onboarding wizard")
    return render_template("onboarding.html",
                          username=session.get('username', 'there'))

@app.route("/history")
def history():
    """Render the chat history page."""
    logger.info("Rendering history page")
    if not session.get('logged_in'):
        return redirect('/login')
        
    # For now, return empty history
    return render_template("history.html", 
                          username=session.get('username', 'User'),
                          email=session.get('email', 'user@example.com'),
                          history=[])

@app.route("/api/chat", methods=["POST"])
def process_chat():
    """Process chat message API endpoint."""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = request.cookies.get("session", "default")
        
        # Get user preferences if provided
        preferences = data.get("preferences", {})
        model_preference = preferences.get("model", "enhanced")
        context_length = preferences.get("context_length", 5)
        topic_filter = preferences.get("topic", "all")
        
        logger.info(f"Processing chat message: {user_message}")
        logger.info(f"User preferences: model={model_preference}, context={context_length}, topic={topic_filter}")
        
        if not user_message:
            return jsonify({
                "response": "Please enter a message to chat with GAKR.",
                "error": "Empty message"
            }), 400
        
        # Add to chat history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        # Add user message to chat history
        chat_histories[session_id].append({
            "type": "user",
            "message": user_message,
            "timestamp": timestamp
        })
        
        # Get limited context history based on preference
        conversation_history = None
        if context_length > 0 and len(chat_histories[session_id]) > 1:
            # Get the limited history (most recent N messages)
            history = chat_histories[session_id]
            conversation_history = history[-context_length:] if len(history) > context_length else history
        
        # Response generation based on user preferences
        if model_preference == "simple":
            # Use basic response generation
            response = generate_simple_response(user_message, session_id)
        else:
            # Use enhanced model with conversation history
            response = generate_simple_response(user_message, session_id)
            
            # Apply topic filter if specified (not "all")
            if topic_filter != "all" and isinstance(response, dict):
                response["topic_filter"] = topic_filter
                # We could implement more specific filtering logic here
        
        # Add timestamp to the response
        if isinstance(response, dict):
            response["timestamp"] = timestamp
            response_text = response["response"]
        else:
            response_text = response
            response = {
                "response": response_text,
                "timestamp": timestamp
            }
        
        # Store bot response in history
        chat_histories[session_id].append({
            "type": "bot",
            "message": response_text,
            "timestamp": timestamp,
            "source": model_preference
        })
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({
            "response": "I'm sorry, I encountered an error while processing your message.",
            "error": str(e)
        }), 500

@app.route("/api/reset", methods=["POST"])
def reset_chat():
    """Reset the chat history."""
    try:
        session_id = request.cookies.get("session", "default")
        if session_id in chat_histories:
            chat_histories[session_id] = []
        return jsonify({"status": "success", "message": "Chat history reset"})
    except Exception as e:
        logger.error(f"Error resetting chat: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error resetting chat: {e}"
        }), 500
        
@app.route("/api/onboarding", methods=["POST"])
def save_onboarding_preferences():
    """Save user onboarding preferences."""
    try:
        data = request.get_json()
        preferences = data.get("preferences", {})
        
        # Store preferences in session
        for key, value in preferences.items():
            session[f"pref_{key}"] = value
            
        # Mark onboarding as completed
        session['onboarding_completed'] = True
        
        # Save the tutorial completion status
        if 'tutorial_completed' in data:
            session['tutorial_completed'] = data['tutorial_completed']
            
        return jsonify({
            "status": "success", 
            "message": "Onboarding preferences saved successfully"
        })
    except Exception as e:
        logger.error(f"Error saving onboarding preferences: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error saving preferences: {e}"
        }), 500

def generate_simple_response(text, session_id="default"):
    """
    Generate a more intelligent response based on the input text.
    This is an improved version using the advanced text processing pipeline and local model inference.
    
    Args:
        text: The user's input text
        session_id: The session identifier for conversation context
    """
    import random
    import datetime
    import re
    
    # Get conversation history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    conversation_history = chat_histories.get(session_id, [])
    
    # Try to use Perplexity API if available
    perplexity_available = False
    
    # Safely check if Perplexity API is available
    try:
        if perplexity_initialized:
            perplexity_available = perplexity_helper.check_perplexity_availability()
    except (ImportError, AttributeError):
        logger.warning("Perplexity helper module not available")
        perplexity_available = False
    
    if perplexity_available and "use_perplexity" not in text.lower():
        try:
            logger.info("Attempting to use Perplexity API")
            # Format conversation history for Perplexity
            formatted_history = []
            
            # Only include the last 5 exchanges for context management
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            
            for entry in recent_history:
                formatted_history.append({
                    "role": "user" if entry["type"] == "user" else "assistant",
                    "content": entry["message"]
                })
            
            perplexity_response = perplexity_helper.generate_perplexity_response(text, formatted_history)
            
            if perplexity_response and not perplexity_response.startswith("I had trouble") and not perplexity_response.startswith("I don't have access"):
                logger.info("Successfully generated response with Perplexity API")
                return perplexity_response
            
            logger.warning("Perplexity API failed or unavailable, falling back to enhanced model")
        except Exception as e:
            logger.error(f"Error using Perplexity API: {e}")
            logger.warning("Falling back to enhanced model")
    
    # Try to use enhanced model if available
    if enhanced_model_available:
        try:
            logger.info("Attempting to use enhanced model")
            # Import the function if it hasn't been imported yet
            try:
                from enhanced_model import generate_enhanced_response
                enhanced_response = generate_enhanced_response(text, conversation_history)
                
                if enhanced_response:
                    logger.info("Successfully generated response with enhanced model")
                    return enhanced_response
            except ImportError:
                logger.warning("Enhanced model module not available")
                
            logger.warning("Enhanced model failed, falling back to neural generator")
        except Exception as e:
            logger.error(f"Error using enhanced model: {e}")
            logger.warning("Falling back to simple model")
    
    # If we're here, try to use the new model inference with text processing pipeline
    try:
        # Import our new text processing pipeline
        from core.model_inference import generate_response as model_generate_response
        
        logger.info(f"Enhanced model not available, falling back to neural generator: '{text[:30]}...'")
        logger.info(f"Chat history available with {len(conversation_history)} messages")
        
        # Log that we're using our neural generator
        logger.info("Using neural generator for response generation")
        
        # Generate response using our model inference
        result = model_generate_response(
            text, 
            conversation_history=conversation_history,
            model_name="simple"  # Use simple mode since enhanced is not available
        )
        
        # Return the response
        if isinstance(result, dict):
            return result
        else:
            return {
                "response": result,
                "analysis_type": "text_generation",
                "sentiment": {
                    "sentiment": "neutral",
                    "score": 0.5
                },
                "confidence": 0.7
            }
    except Exception as e:
        logger.error(f"Error using neural generator: {e}")
        # Continue with the original code for the fallback
    
    # Normalize text for easier matching
    normalized_text = text.lower().strip()
    
    # Check if it's a question
    is_question = any(normalized_text.startswith(q) for q in ["what", "who", "when", "where", "why", "how", "is", "are", "can", "could", "will", "would"]) or "?" in normalized_text
    
    # Simple sentiment analysis with expanded vocabulary
    positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "wonderful", "awesome", "fantastic", 
                      "nice", "pleased", "glad", "joy", "exciting", "brilliant", "terrific", "perfect", "impressive", "beautiful"]
    negative_words = ["bad", "terrible", "awful", "hate", "poor", "sad", "horrible", "dislike", "disappointed", 
                      "upset", "angry", "frustrating", "annoying", "useless", "stupid", "boring", "worse", "worst", "not satisfied"]
    
    words = normalized_text.split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    sentiment = "NEUTRAL"
    sentiment_score = 0.5
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
        sentiment_score = 0.5 + (0.1 * positive_count)
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
        sentiment_score = 0.5 - (0.1 * negative_count)
    
    # Cap sentiment score between 0 and 1
    sentiment_score = max(0, min(1, sentiment_score))
    
    # Check for math operations
    math_pattern = re.compile(r'(\d+)\s*([\+\-\*\/])\s*(\d+)')
    math_match = math_pattern.search(normalized_text)
    
    if math_match:
        try:
            num1, op, num2 = math_match.groups()
            num1 = float(num1)
            num2 = float(num2)
            
            result = None
            if op == '+':
                result = knowledge_base["math"]["addition"](num1, num2)
                op_name = "addition"
            elif op == '-':
                result = knowledge_base["math"]["subtraction"](num1, num2)
                op_name = "subtraction"
            elif op == '*':
                result = knowledge_base["math"]["multiplication"](num1, num2)
                op_name = "multiplication"
            elif op == '/':
                result = knowledge_base["math"]["division"](num1, num2)
                op_name = "division"
            
            if result is not None:
                # Format result to remove trailing zeros for integers
                if result == int(result):
                    result = int(result)
                return {
                    "response": f"The result of {num1} {op} {num2} is {result}.",
                    "sentiment": {
                        "sentiment": sentiment,
                        "score": sentiment_score,
                        "analysis_type": "sentiment"
                    },
                    "analysis_type": "calculation"
                }
        except Exception as e:
            logger.error(f"Error processing math operation: {e}")
    
    # Check for "means" or "definition" type queries
    means_pattern = re.compile(r'(\w+)\s+means', re.IGNORECASE)
    means_match = means_pattern.search(normalized_text)
    
    if means_match:
        term = means_match.group(1).lower()
        if term in knowledge_base and isinstance(knowledge_base[term], str):
            return {
                "response": knowledge_base[term],
                "sentiment": {
                    "sentiment": sentiment,
                    "score": sentiment_score,
                    "analysis_type": "sentiment"
                },
                "analysis_type": "knowledge_base"
            }
    
    # Check knowledge base for specific topics
    for key, info in knowledge_base.items():
        if key != "math" and key in normalized_text:
            if isinstance(info, str):
                return {
                    "response": info,
                    "sentiment": {
                        "sentiment": sentiment,
                        "score": sentiment_score,
                        "analysis_type": "sentiment"
                    },
                    "analysis_type": "knowledge_base"
                }
    
    # Topic detection - expanded implementation
    topics = {
        "identity": ["who are you", "what are you", "your name", "what is your name", "about yourself", "tell me about you", "name"],
        "capabilities": ["what can you do", "your abilities", "your features", "what do you do", "help me with", "how do you work"],
        "greetings": ["hello", "hi ", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"],
        "time": ["time", "date", "day", "month", "year", "today", "tomorrow", "yesterday"],
        "jokes": ["joke", "funny", "laugh", "humor", "comedy"],
        "technology": ["computer", "programming", "code", "software", "hardware", "app", "website", "internet", "html", "css", "javascript", "tech", "technology"],
        "personal": ["my name is", "i am", "i'm", "my name", "myself", "about me"],
        "feedback": ["thanks", "thank you", "appreciate", "helpful", "not helpful", "useless", "satisfied", "not satisfied", "good job", "bad job"],
        "education": ["btech", "mtech", "phd", "degree", "college", "university", "school", "education", "student", "study", "academics", "course"],
        "engineering": ["engineering", "engineer", "cse", "ece", "electrical"],
        "business": ["business", "company", "startup", "entrepreneur", "commerce", "market", "industry", "finance", "economics"],
        "entertainment": ["movie", "film", "music", "song", "game", "play", "show", "entertainment", "book", "novel", "story"],
        "symbols": ["@", "#", "&", "symbol", "sign", "character", "emoji"],
        "animals": ["animal", "pet", "dog", "cat", "bird", "fish", "creature", "wildlife", "species"],
        "information": ["what is", "define", "definition", "meaning", "tell me about", "explain", "what are", "what does", "how is", "means"]
    }
    
    detected_topics = []
    for topic, keywords in topics.items():
        if any(keyword in normalized_text for keyword in keywords):
            detected_topics.append(topic)
    
    # Generate appropriate response based on detected topics and question type
    if "identity" in detected_topics:
        responses = [
            "I am GAKR, an AI chatbot built to process and analyze text without external API dependencies.",
            "My name is GAKR. I'm a chatbot designed to help answer questions and have conversations.",
            "I'm GAKR, a text-based AI assistant that works completely locally, without calling external APIs.",
            "GAKR is my name - it stands for a chatbot that analyses and responds to text using pre-trained models."
        ]
        response = random.choice(responses)
        
    elif "capabilities" in detected_topics:
        responses = [
            "I can analyze sentiment, answer questions, and have conversations with you. I'm designed to work without external API dependencies.",
            "I can help with answering questions, analyzing the sentiment of your messages, and engaging in conversation. I'm completely self-contained.",
            "My capabilities include natural language understanding, sentiment analysis, and generating contextual responses to your questions.",
            "I can assist with information, engage in conversation, detect the emotional tone of your messages, and provide helpful responses."
        ]
        response = random.choice(responses)
        
    elif "greetings" in detected_topics:
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What would you like to chat about?",
            "Greetings! I'm GAKR, ready to assist you.",
            "Hey! How can I be of service today?"
        ]
        response = random.choice(responses)
        
    elif "time" in detected_topics:
        now = datetime.datetime.now()
        
        if "year" in normalized_text:
            response = f"The current year is {now.year}."
        elif "month" in normalized_text:
            response = f"The current month is {now.strftime('%B')}."
        elif "day" in normalized_text:
            response = f"Today is {now.strftime('%A')}."
        elif "date" in normalized_text or "today" in normalized_text:
            response = f"Today's date is {now.strftime('%B %d, %Y')}."
        elif "time" in normalized_text:
            response = f"The current time is {now.strftime('%H:%M:%S')}."
        else:
            response = f"Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%H:%M:%S')}."
    
    elif "jokes" in detected_topics:
        jokes = [
            "Why did the AI assistant go to art school? To learn how to draw conclusions!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the chatbot cross the road? To get to the other website!",
            "I asked the AI assistant for a joke about construction, but it's still working on it.",
            "Why was the computer cold? It left its Windows open!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "Why do programmers prefer dark mode? Because light attracts bugs!"
        ]
        response = random.choice(jokes)
    
    elif "technology" in detected_topics:
        # Check for specific technology terms
        tech_terms = ["html", "css", "javascript", "computer", "programming", 
                      "code", "software", "hardware", "app", "website", "internet"]
        
        for term in tech_terms:
            if term in normalized_text:
                response = knowledge_base.get(term, "That's an interesting technology topic!")
                break
        else:
            response = "I can provide information about various technology topics. Could you be more specific about what you'd like to know?"
    
    elif "personal" in detected_topics:
        name_match = re.search(r"my name is (\w+)", normalized_text)
        if name_match:
            name = name_match.group(1).capitalize()
            responses = [
                f"Nice to meet you, {name}! How can I help you today?",
                f"Hello {name}! I'm GAKR. What would you like to chat about?",
                f"It's great to know your name, {name}. Is there something specific you'd like to talk about?",
                f"Thanks for introducing yourself, {name}. How can I assist you?"
            ]
            response = random.choice(responses)
        else:
            response = "It's nice to learn more about you. Is there something specific I can help you with?"
            
    elif "education" in detected_topics:
        # Check if the text contains specific education terms in our knowledge base
        education_terms = ["btech", "mtech", "phd", "computer science", "cse"]
        for term in education_terms:
            if term in normalized_text:
                response = knowledge_base.get(term, "That's an interesting education topic. I can provide information about various degrees and educational programs if you have specific questions.")
                break
        else:
            responses = [
                "Education is a valuable investment in your future. What specific aspect of education would you like to discuss?",
                "I can provide information about various degree programs and educational paths. What would you like to know more about?",
                "Education comes in many forms, from formal degrees to self-directed learning. What specific educational topic interests you?",
                "Learning is a lifelong journey. Can you specify which educational topic you'd like information about?"
            ]
            response = random.choice(responses)
    
    elif "engineering" in detected_topics:
        # Check if the text contains specific engineering terms in our knowledge base
        engineering_terms = ["cse", "ece", "electrical engineering", "computer engineering", "software engineering", "data engineering"]
        for term in engineering_terms:
            if term in normalized_text:
                response = knowledge_base.get(term, "Engineering is a diverse field with many specializations. I can provide more specific information if you name a particular branch of engineering.")
                break
        else:
            responses = [
                "Engineering involves applying scientific and mathematical principles to design and build solutions. What specific branch interests you?",
                "There are many branches of engineering, from civil and mechanical to software and biomedical. Which would you like to learn about?",
                "Engineering is about solving problems through technical innovation. What particular aspect of engineering are you curious about?",
                "Engineers create, design, and improve structures, machines, systems, and processes. Which engineering field would you like to explore?"
            ]
            response = random.choice(responses)
            
    elif "business" in detected_topics:
        # Check if the text contains specific business terms in our knowledge base
        business_terms = ["business", "entrepreneurship", "e-commerce", "digital marketing"]
        for term in business_terms:
            if term in normalized_text:
                response = knowledge_base.get(term, "Business encompasses many aspects of commerce and enterprise. I can provide more specific information about different business topics.")
                break
        else:
            responses = [
                "Business covers a wide range of commercial activities and enterprises. What specific aspect would you like to know more about?",
                "From startups to corporations, business takes many forms. What particular business topic interests you?",
                "Business involves the organized efforts of individuals to produce and sell goods and services. What specific area would you like to discuss?",
                "The world of business includes entrepreneurship, management, marketing, and finance. Which aspect are you curious about?"
            ]
            response = random.choice(responses)
    
    elif "entertainment" in detected_topics:
        # Check if the text contains specific entertainment terms in our knowledge base
        entertainment_terms = ["movie", "music", "game", "book", "library"]
        for term in entertainment_terms:
            if term in normalized_text:
                response = knowledge_base.get(term, "Entertainment comes in many forms, from movies and music to books and games. I can provide more specific information if you ask about a particular form of entertainment.")
                break
        else:
            responses = [
                "Entertainment encompasses various activities that provide enjoyment and amusement. What specific form interests you?",
                "From movies and music to books and games, entertainment takes many forms. What would you like to know more about?",
                "Entertainment is a broad category covering activities that engage and divert the mind. Which specific type would you like to discuss?",
                "The world of entertainment offers countless ways to relax and enjoy leisure time. What particular area are you curious about?"
            ]
            response = random.choice(responses)
    
    elif "symbols" in detected_topics:
        # Check if the text contains specific symbols in our knowledge base
        symbols = ["@", "#", "&"]
        for symbol in symbols:
            if symbol in normalized_text:
                response = knowledge_base.get(symbol, f"The symbol '{symbol}' has various meanings depending on context. In different fields like programming, mathematics, or social media, it serves different purposes.")
                break
        else:
            responses = [
                "Symbols are visual representations that convey meaning across languages and cultures. Which symbol are you asking about?",
                "From mathematical notation to keyboard characters, symbols have various uses and meanings. Can you specify which symbol you're interested in?",
                "Symbols can represent complex ideas in simple visual forms. Is there a particular symbol you'd like information about?",
                "Many symbols have rich histories and multiple meanings across different contexts. Which one would you like to learn about?"
            ]
            response = random.choice(responses)
    
    elif "animals" in detected_topics:
        responses = [
            "The animal kingdom is incredibly diverse, with millions of species from microscopic organisms to massive mammals. What specific animals interest you?",
            "Animals come in countless forms and inhabit virtually every ecosystem on Earth. Which animals would you like to learn more about?",
            "From domesticated pets to wild creatures, animals play important roles in our world. What particular animal or animal group are you curious about?",
            "The study of animals, zoology, reveals fascinating behaviors, adaptations, and relationships. Which aspect of animals would you like to explore?"
        ]
        response = random.choice(responses)
    
    elif "feedback" in detected_topics:
        if sentiment == "POSITIVE":
            responses = [
                "I'm glad to hear that! Thank you for your positive feedback.",
                "I appreciate your kind words! Is there anything else I can help with?",
                "Thank you for the positive feedback. I aim to be helpful!",
                "It's great to know I could assist you. What else would you like to talk about?"
            ]
            response = random.choice(responses)
        elif sentiment == "NEGATIVE":
            responses = [
                "I'm sorry I couldn't meet your expectations. I'm still learning and improving.",
                "I apologize if my responses weren't helpful. Could you tell me how I can do better?",
                "I understand your frustration. I'm working to improve my capabilities.",
                "Thank you for your feedback. I'll try to provide better assistance in the future."
            ]
            response = random.choice(responses)
        else:
            response = "Thank you for your feedback. Is there anything specific you'd like me to help with?"
    
    elif is_question:
        # Check for definition/explanation questions
        definition_pattern = re.compile(r'what(?:\s+is|\'s|\s+are|\s+does)(?:\s+the)?(?:\s+a)?\s+([a-z\s]+)', re.IGNORECASE)
        definition_match = definition_pattern.search(normalized_text)
        
        if definition_match:
            topic = definition_match.group(1).strip().lower()
            if topic in knowledge_base:
                return {
                    "response": knowledge_base[topic],
                    "sentiment": {
                        "sentiment": sentiment,
                        "score": sentiment_score,
                        "analysis_type": "sentiment"
                    },
                    "analysis_type": "knowledge_base"
                }
                
        # Handle specific question types
        if normalized_text.startswith("how"):
            if "how are you" in normalized_text:
                responses = [
                    "I'm functioning well, thank you for asking! How can I assist you today?",
                    "I'm doing great! Thank you for asking. How about you?",
                    "All systems operational! How may I help you?",
                    "I'm well, thanks! What can I do for you today?"
                ]
                response = random.choice(responses)
            else:
                responses = [
                    "That's a good 'how' question. While I have limited knowledge, I'll do my best to help.",
                    "Interesting question about process or method. I can provide basic guidance on this topic.",
                    "That's a question about method or process. I can offer some thoughts, but may not have complete expertise.",
                    "Good question about 'how'. I'll try to provide a helpful response based on my knowledge."
                ]
                response = random.choice(responses)
        
        elif normalized_text.startswith("what"):
            if any(phrase in normalized_text for phrase in ["what is", "what are", "what does"]):
                # Check for specific topics in our knowledge base
                for key, value in knowledge_base.items():
                    if isinstance(value, str) and key in normalized_text:
                        response = value
                        break
                else:
                    responses = [
                        "That's an interesting question about definitions or explanations. I'll try to help with what I know.",
                        "You're asking for information or an explanation. Let me share what I understand about this.",
                        "That's a question seeking information. I can provide a basic answer based on my knowledge.",
                        "You're asking about what something is or means. I'll do my best to explain based on my understanding."
                    ]
                    response = random.choice(responses)
            else:
                responses = [
                    "That's a good 'what' question. I'll try to provide information based on my knowledge.",
                    "You're asking for specific information. I'll do my best to help with what I know.",
                    "Interesting question about 'what'. I can provide some thoughts on this topic.",
                    "That's a question seeking details. I'll share what I understand about this."
                ]
                response = random.choice(responses)
        
        elif normalized_text.startswith("when"):
            responses = [
                "That's a question about timing. I don't have complete information, but I can offer some thoughts.",
                "You're asking about when something happens. I don't have real-time data, but can provide general information.",
                "That's a 'when' question. I can give you some general guidance, though I don't have access to specific schedules.",
                "Interesting question about timing. I'll share what I understand about this, though my knowledge is limited."
            ]
            response = random.choice(responses)
        
        elif normalized_text.startswith("where"):
            responses = [
                "That's a question about location. I don't have access to maps or real-time location data, but can offer general knowledge.",
                "You're asking about where something is. I can provide general information, but not specific current locations.",
                "That's a 'where' question. I'll try to help with general knowledge, though I don't have access to location services.",
                "Interesting question about places. I can share general information, but not specific directions or current locations."
            ]
            response = random.choice(responses)
        
        elif normalized_text.startswith("why"):
            responses = [
                "That's an interesting question about reasons or causes. I'll try to provide some perspective.",
                "You're asking about why something happens. I can offer some thoughts based on my knowledge.",
                "That's a 'why' question seeking explanations. I'll share what I understand about this.",
                "Interesting question about causes. I'll try to provide insight based on general principles."
            ]
            response = random.choice(responses)
        
        elif normalized_text.startswith(("is", "are", "can", "could", "will", "would")):
            responses = [
                "That's a yes/no type question. I can provide some thoughts, but may not have a definitive answer.",
                "You're asking for confirmation or possibility. I'll share what I understand about this topic.",
                "That's a question seeking verification. I can offer some perspective based on my knowledge.",
                "Interesting question about possibilities. I'll try to provide a thoughtful response."
            ]
            response = random.choice(responses)
        
        else:
            # General question responses
            responses = [
                "That's an interesting question. While I don't have all the answers, I'll try to provide a helpful response.",
                "I understand you're asking about that topic. I'll share what I know, though my knowledge is limited.",
                "Good question! I'll try my best to give you a useful answer based on my understanding.",
                "I'm analyzing your question. I can provide some thoughts on this, though I may not have complete information."
            ]
            response = random.choice(responses)
    
    else:
        # Handle non-question inputs based on sentiment
        if sentiment == "POSITIVE":
            responses = [
                "I'm glad to hear that! Your message seems positive.",
                "That sounds great! I detect a positive sentiment in your message.",
                "Wonderful! I sense you're feeling good about this.",
                "Excellent! Your message has a positive tone to it."
            ]
            response = random.choice(responses)
        elif sentiment == "NEGATIVE":
            responses = [
                "I'm sorry to hear that. Your message seems to have a negative tone.",
                "I understand this might be frustrating for you. I detect some concern in your message.",
                "I see this is something that's bothering you. Let me try to help.",
                "I notice your message has a negative sentiment. How can I assist with this situation?"
            ]
            response = random.choice(responses)
        else:
            # More varied neutral responses
            responses = [
                "Thank you for your message. I'm here to assist you with any questions you might have.",
                "I understand. Please let me know if there's something specific I can help with.",
                "I'm processing what you've shared. Is there anything particular you'd like to discuss?",
                "I see what you're saying. What would you like to talk about next?",
                "Thanks for sharing that. What else would you like to discuss?",
                "I'm listening. Is there something specific you're interested in learning about?",
                "I appreciate your input. How else can I assist you today?",
                "I'm here to chat about various topics. What would you like to explore next?"
            ]
            response = random.choice(responses)
    
    # We're using our knowledge-based approach for responses
    # No Kaggle model integration due to disk quota limitations
    
    # Try to use the enhanced model if available, otherwise fall back to simple neural generator
    try:
        # First try the enhanced model
        try:
            # Import lazily to avoid unnecessary loading during startup
            from enhanced_model import generate_enhanced_response
            
            # Get conversation history for context
            conversation_history = None
            if session_id in chat_histories and len(chat_histories[session_id]) > 0:
                conversation_history = []
                for item in chat_histories[session_id][-5:]:  # Use last 5 messages for context
                    conversation_history.append({
                        "role": "user",
                        "content": item["user_message"]
                    })
                    conversation_history.append({
                        "role": "assistant",
                        "content": item["response"]
                    })
                
                logger.info(f"Using conversation history with {len(conversation_history)} messages for enhanced model")
            
            # Generate response using enhanced model
            enhanced_response = generate_enhanced_response(text, conversation_history)
            
            if enhanced_response:
                logger.info("Using enhanced model for response generation")
                response = enhanced_response
                return {
                    "response": response,
                    "sentiment": {
                        "sentiment": sentiment,
                        "score": sentiment_score,
                        "analysis_type": "sentiment"
                    },
                    "analysis_type": "enhanced_model"
                }
        except Exception as e:
            logger.info(f"Enhanced model not available, falling back to neural generator: {e}")
            
        # Fall back to neural generator
        from simple_neural import generate_neural_response
        
        # Get conversation history for context
        if session_id in chat_histories and len(chat_histories[session_id]) > 0:
            # The neural generator doesn't use history directly, but we log it for future enhancements
            logger.info(f"Chat history available with {len(chat_histories[session_id])} messages")
        
        # Generate response using neural generator
        neural_response_text = generate_neural_response(text)
        
        if neural_response_text:
            logger.info("Using neural generator for response generation")
            response = neural_response_text
    except Exception as e:
        logger.info(f"Using rule-based response generation: {e}")
        # Continue with the knowledge-based approach
    
    return {
        "response": response,
        "sentiment": {
            "sentiment": sentiment,
            "score": sentiment_score,
            "analysis_type": "sentiment"
        },
        "analysis_type": "text_generation" if not is_question else "question_answering"
    }

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
