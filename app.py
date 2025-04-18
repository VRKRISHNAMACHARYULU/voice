from flask import Flask, request, jsonify
import json
import numpy as np
from flask_cors import CORS
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'similarity_threshold': 0.5,
    'port': 5000,
    'host': '0.0.0.0'
}

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("The 'sentence_transformers' library is not installed. Install it using 'pip install sentence-transformers'.")
    raise ImportError("The 'sentence_transformers' library is not installed. Install it using 'pip install sentence-transformers'.")

from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://myportfolio-liard-two-33.vercel.app",  # Your Vercel domain
    "http://localhost:3000",  # For local development
    # Add any other domains you need
]}})

logger.info("Loading model and data...")
try:
    # Load the data
    with open('qa_data.json', 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    qa_dict = data['qa_dict']
    
    # Load embeddings from numpy file
    embeddings = np.load('question_embeddings.npy')
    
    # Load the saved model
    try:
        model = SentenceTransformer('sentence_transformer_model')
        logger.info("Loaded saved model successfully")
    except Exception as e:
        logger.warning(f"Could not load saved model, using default instead: {e}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load fallbacks
    with open("fallbacks.txt", "r") as f:
        fallbacks = [line.strip() for line in f.readlines()]
    
    logger.info("Model and data loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or data: {e}")
    exit(1)

@app.route('/chat', methods=['POST'])
def ask():
    try:
        # Get the question from the request
        data = request.json
        if not data or 'question' not in data:
            logger.warning("Request missing 'question' field")
            return jsonify({'error': 'No question provided'}), 400
        
        query = data['question']
        
        # Special greetings
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().strip() in greetings:
            return jsonify({
                'answer': "Hi! I'm your portfolio assistant. Feel free to ask about my skills, experience, or projects!",
                'confidence': 1.0
            })
        
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
        else:
            answer = random.choice(fallbacks)
        
        return jsonify({
            'answer': answer,
            'confidence': float(best_score),
            'matched_question': questions[best_idx] if best_score >= threshold else None
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'questions_count': len(questions) if 'questions' in locals() else 0
    })

if __name__ == '__main__':
    # Set debug=False for production
    logger.info(f"Starting API server on port {CONFIG['port']}...")
    app.run(debug=False, host=CONFIG['host'], port=CONFIG['port'])
