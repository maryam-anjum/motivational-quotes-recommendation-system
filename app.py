from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tokenizer import tokenizer 

app = Flask(__name__)

# Load the dataframe and vectorizers
df = pd.read_pickle('quotes_df.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
count_vectorizer = joblib.load('count_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')
category_matrix = joblib.load('category_matrix.joblib')

# Define the recommendation function
def recommend_quotes_by_category(query_category, top_n=5, random_sample_size=None):
    query_vector = count_vectorizer.transform([query_category])
    category_similarity = cosine_similarity(query_vector, category_matrix).flatten()
    similar_category_indices = category_similarity.argsort()[-top_n:][::-1]
    top_similar_quotes = df.iloc[similar_category_indices]['quote'].tolist()
    random.shuffle(top_similar_quotes)  # Shuffle the quotes
    if random_sample_size is not None:
        if len(top_similar_quotes) > random_sample_size:

            top_similar_quotes = random.sample(top_similar_quotes, random_sample_size)
    return top_similar_quotes

@app.route('/')
def home():
    return render_template('edu.html')

@app.route('/get_quote', methods=['POST'])
def get_quote():
    try:
        data = request.get_json()
        user_input = data['mood']
        recommended_quotes = recommend_quotes_by_category(user_input)
        return jsonify({'quotes': recommended_quotes})
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
