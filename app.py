from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from src.main import sent_model, act_model

app = Flask(__name__)

# Load your pre-trained models (sentiment_model, actionability_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    # Get the review from the query parameter
    review = request.args.get('review', '')

    if not review:
        return "No review provided", 400

    # Use the models to make predictions
    sentiment_prediction = sentiment_model.predict([review])[0]
    actionability_prediction = actionability_model.predict([review])[0]

    # Map predictions to readable labels
    sentiment_label = "Positive" if sentiment_prediction == 1 else "Negative"
    actionability_label = f"{actionability_prediction:.2f}"  # Assuming it's a score

    # Pass predictions to the template
    return render_template('result.html', review=review, sentiment=sentiment_label, actionability=actionability_label)

def tokenize_and_embed(text_data):
    """
    Tokenizes the text data and converts it to word embeddings using SpaCy.
    Args:
        text_data (list): A list of text strings to be processed.
    Returns:
        list: A list of lists containing embeddings for each token in each document.
    """
    docs = list(text_to_nlp.pipe(text_data))
    embeddings = [[token.vector for token in doc] for doc in docs]
    return embeddings

def standardize_length(embeddings):
    """
    Ensures all embedding lists are the same length by padding shorter ones with zero vectors.
    Args:
        embeddings (list): A list of lists of embeddings.
    Returns:
        list: A list of lists with padded embeddings to ensure uniform length.
    """
    max_length = max(len(tokens) for tokens in embeddings)
    embedding_dim = len(embeddings[0][0]) if embeddings[0] else 0
    padded_embeddings = [[np.zeros(embedding_dim)] * (max_length - len(tokens)) + tokens for tokens in embeddings]
    return padded_embeddings

def convert_to_array(padded_embeddings):
    """
    Converts a list of padded embeddings into a numpy array.
    Args:
        padded_embeddings (list): A list of lists of padded embeddings.
    Returns:
        numpy.ndarray: A numpy array containing the embeddings suitable for machine learning input.
    """
    return np.array(padded_embeddings)


# Load your models
sentiment_model = sent_model  # Load or pass your trained sentiment_model here
actionability_model = act_model  # Load or pass your trained actionability_model here

@app.route("/")
def index():
    # Render the HTML frontend
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get data sent from the frontend
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X_embeddings = tokenize_and_embed(text)  # Tokenize and get embeddings  
    X_padded = standardize_length(X_embeddings)  # Standardize lengths
    X = convert_to_array(X_padded)
    text_array = np.array([text_padded])  # Convert to a numpy array

    sentiment_pred = sentiment_model.predict(text_array)[0][0]  # Get prediction for sentiment
    actionability_pred = actionability_model.predict(text_array)[0][0]  # Get prediction for actionability

    return jsonify({
        "sentiment": "POSITIVE" if sentiment_pred > 0.5 else "NEGATIVE",
        "actionability_score": round(actionability_pred, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
