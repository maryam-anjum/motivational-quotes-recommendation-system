# app.py
from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # Adjust num_labels as needed

# Load the model weights from the .pth file
model_path = 'motivational_quotes_model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    logger.info(f'Model loaded successfully from {model_path}')
except Exception as e:
    logger.error(f'Error loading the model: {e}')

# Example quotes data, should be replaced with actual quotes
quotes_data = {
    0: ["Take responsibility for your own happiness. Do not expect people or things to bring you happiness, or you could be disappointed.", "Rich in happiness is about choosing new perspectives, new habits, and a new emotional future."],
    1: ["The future is as bright as the promises of God.", "To a father when a child dies, the future dies; to a child when a parent dies, the past dies."],
    2: ["You will not be able to find peace if you are the source of unhappiness for others.", "Trust your own instincts; go inside, follow your heart right from the start. Go ahead and stand up for what you believe in. As I've learned, that's the path to happiness."],
    3: ["No one person can possibly combine all the elements supposed to make up what everyone means by friendship.", "No person is your friend who demands your silence or denies your right to grow."]
}

# Define categories based on your model
categories = ['happy', 'hurt', 'inspirational', 'friendship']

def get_recommendation(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    recommended_quotes = generate_quotes(predicted_class)
    return recommended_quotes

def generate_quotes(predicted_class):
    quotes = quotes_data.get(predicted_class, ["No quotes available."])
    return quotes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_quote', methods=['POST'])
def get_quote():
    try:
        data = request.get_json()
        user_input = data['mood']
        recommended_quotes = get_recommendation(user_input)
        return jsonify({'quotes': recommended_quotes})
    except Exception as e:
        logger.error(f'Error processing the quote request: {e}')
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
