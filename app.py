from flask import Flask, request, jsonify, render_template
import random
from src.main import sent, action

app = Flask(__name__)

# Mock function to classify sentiment, actionability, and topic
def analyze_feedback(review):
    sentiment = sent()  # Replace with sentiment model
    actionability_score = action()  # Replace with actionability model

    return sentiment, actionability_score, ""


@app.route("/")
def index():
    # Serve index.html from the root folder
    return send_file("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    review = data.get("review", "")
    sentiment, actionability_score, topic = analyze_feedback(review)
    return jsonify({
        "sentiment": sentiment,
        "actionability_score": actionability_score,
        "topic": topic
    })

if __name__ == "__main__":
    app.run(debug=True)
