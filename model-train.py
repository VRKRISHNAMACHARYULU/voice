import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import joblib
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading training data...")
try:
    df = pd.read_excel("training_data.xlsx")
    logger.info(f"Loaded {len(df)} question-answer pairs")
except Exception as e:
    logger.error(f"Error loading training data: {e}")
    exit(1)

# Clean and prepare data
logger.info("Cleaning data...")
required_columns = ["questions", "answers"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logger.error(f"Error: Excel file missing required columns: {', '.join(missing_columns)}")
    logger.error("Please ensure your Excel file has 'questions' and 'answers' columns")
    exit(1)

df = df.dropna()
df = df[df['questions'].str.strip() != '']
df = df[df['answers'].str.strip() != '']

# Create dictionary for easy lookup
qa_dict = dict(zip(df['questions'], df['answers']))

# Initialize a pre-trained sentence transformer model
logger.info("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small but effective model

# Generate embeddings for all questions
logger.info("Generating embeddings...")
questions = df['questions'].tolist()
question_embeddings = model.encode(questions)

# Create a simple semantic search function
def find_best_answer(query, embeddings, questions, qa_dict, threshold=0.6):
    # Encode the query
    query_embedding = model.encode([query])[0]
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    # Check if the best match is good enough
    if best_score >= threshold:
        return qa_dict[questions[best_idx]], best_score
    else:
        return None, best_score

# Save the model and data
logger.info("Saving model and embeddings...")
data_package = {
    'questions': questions,
    'qa_dict': qa_dict
}
with open('qa_data.json', 'w') as f:
    json.dump(data_package, f)

# Save embeddings separately as numpy array
np.save('question_embeddings.npy', question_embeddings)

# Save the full model
os.makedirs('sentence_transformer_model', exist_ok=True)
torch.save(model.state_dict(), 'sentence_transformer_model/pytorch_model.bin')
with open('sentence_transformer_model/config.json', 'w') as f:
    json.dump({'model_type': 'SentenceTransformer', 'model_name': 'all-MiniLM-L6-v2'}, f)

# Create fallback responses
fallbacks = [
    "I don't have that information in my training data.",
    "I'm not sure about that. Would you like to know something else about my portfolio?",
    "I don't have details on that yet. Please ask something about my skills, projects, or background.",
    "I can't answer that. Feel free to ask about my work experience or technical skills instead.",
    "That's beyond my current knowledge. Let me tell you about my portfolio instead."
]
with open("fallbacks.txt", "w") as f:
    for fallback in fallbacks:
        f.write(fallback + "\n")

logger.info("✅ Model, embeddings, and fallbacks saved successfully.")

# Test the model
logger.info("\nTesting the model with a few examples:")
test_questions = [
    "What is your name?",
    "Tell me about your internship",
    "What programming languages do you know?",
    "Where did you go to college?",
    "This is a completely unrelated question that shouldn't match anything"
]

for question in test_questions:
    answer, score = find_best_answer(question, question_embeddings, questions, qa_dict)
    logger.info(f"Q: {question}")
    logger.info(f"A: {answer if answer else 'No good match found'} (score: {score:.2f})")
    logger.info("-" * 40)

logger.info("\nRecommendations for improving accuracy:")
logger.info("1. Add more diverse questions to your training data")
logger.info("2. Include multiple variations of the same question")
logger.info("3. Fine-tune the similarity threshold based on your testing")