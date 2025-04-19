from flask import Flask, request, jsonify, make_response
import json
import numpy as np
from flask_cors import CORS
import logging
import os
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'similarity_threshold': 0.5,
    'port': int(os.environ.get('PORT', 5000)),
    'host': '0.0.0.0'
}

# List of allowed origins
ALLOWED_ORIGINS = [
    "https://myportfolio-liard-two-33.vercel.app",
    "http://localhost:3000",
    # Add any other domains you need
]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("The 'sentence_transformers' library is not installed. Install it using 'pip install sentence-transformers'.")
    raise ImportError("The 'sentence_transformers' library is not installed. Install it using 'pip install sentence-transformers'.")

app = Flask(__name__)

# Configure CORS with explicit options
CORS(app, 
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

logger.info("Loading model and data...")
try:
    # Load the data
    with open('qa_data.json', 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    qa_dict = data['qa_dict']
    
    # Load embeddings from numpy file if it exists, otherwise use from JSON
    try:
        embeddings = np.load('question_embeddings.npy')
        logger.info("Loaded embeddings from .npy file")
    except FileNotFoundError:
        logger.info("Embeddings .npy file not found, using from JSON")
        embeddings = np.array(data.get('embeddings', []))
        # Save for next time
        np.save('question_embeddings.npy', embeddings)
    
    # Load the saved model
    try:
        model = SentenceTransformer('sentence_transformer_model')
        logger.info("Loaded saved model successfully")
    except Exception as e:
        logger.warning(f"Could not load saved model, using default instead: {e}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load fallbacks
    try:
        with open("fallbacks.txt", "r") as f:
            fallbacks = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logger.warning("Fallbacks file not found, using default fallbacks")
        fallbacks = [
            "I'm not sure how to answer that. Could you ask about my skills or projects instead?",
            "I don't have information about that. Feel free to ask about my professional experience.",
            "That's beyond my current knowledge. I'd be happy to tell you about my portfolio."
        ]
    
    logger.info(f"Model and data loaded successfully! Loaded {len(questions)} questions.")
except Exception as e:
    logger.error(f"Error loading model or data: {e}")
    logger.error(traceback.format_exc())
    exit(1)

# Helper function to add CORS headers
def add_cors_headers(response, origin):
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Origin', ALLOWED_ORIGINS[0])
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/chat', methods=['OPTIONS'])
def chat_options():
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    response = make_response('')
    response.status_code = 204  # No content
    return add_cors_headers(response, origin)

@app.route('/chat', methods=['POST'])
def ask():
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    
    try:
        # Get the question from the request
        data = request.json
        if not data or 'question' not in data:
            logger.warning("Request missing 'question' field")
            response = jsonify({'error': 'No question provided'})
            response.status_code = 400
            return add_cors_headers(response, origin)
        
        query = data['question']
        logger.info(f"Received question: {query}")
        
        # Special greetings
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().strip() in greetings:
            response = jsonify({
                'answer': "Hi! I'm your portfolio assistant. Feel free to ask about my skills, experience, or projects!",
                'confidence': 1.0
            })
            return add_cors_headers(response, origin)
        
        # Process the question
        query_embedding = model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Check if the best match is good enough
        threshold = CONFIG['similarity_threshold']
        if best_score >= threshold:
            answer = qa_dict[questions[best_idx]]
            logger.info(f"Found match: '{questions[best_idx]}' with confidence {best_score:.4f}")
        else:
            answer = random.choice(fallbacks)
            logger.info(f"No good match found. Best match was '{questions[best_idx]}' with low confidence {best_score:.4f}")
        
        response = jsonify({
            'answer': answer,
            'confidence': float(best_score),
            'matched_question': questions[best_idx] if best_score >= threshold else None
        })
        
        return add_cors_headers(response, origin)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        response = jsonify({'error': f'Server error: {str(e)}'})
        response.status_code = 500
        return add_cors_headers(response, origin)

@app.route('/health', methods=['GET'])
def health_check():
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    response = jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'questions_count': len(questions) if 'questions' in globals() else 0,
        'cors_allowed_origins': ALLOWED_ORIGINS
    })
    return add_cors_headers(response, origin)

@app.route('/', methods=['GET'])
def home():
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    response = jsonify({
        'message': 'Portfolio Assistant API is running',
        'endpoints': {
            '/chat': 'POST - Send questions to the assistant',
            '/health': 'GET - Check system health'
        }
    })
    return add_cors_headers(response, origin)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    response = jsonify({'error': 'Endpoint not found'})
    response.status_code = 404
    return add_cors_headers(response, origin)

@app.errorhandler(405)
def method_not_allowed(e):
    origin = request.headers.get('Origin', ALLOWED_ORIGINS[0])
    response = jsonify({'error': 'Method not allowed'})
    response.status_code = 405
    return add_cors_headers(response, origin)

if __name__ == '__main__':
    # Set debug=False for production
    logger.info(f"Starting API server on {CONFIG['host']}:{CONFIG['port']}...")
    app.run(debug=False, host=CONFIG['host'], port=CONFIG['port'])
