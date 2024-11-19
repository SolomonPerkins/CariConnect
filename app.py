from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Load models at startup
try:
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        models = {
            'tfidf': pickle.load(f)
        }
    with open('models/maxabs_scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    with open('models/svd_model.pkl', 'rb') as f:
        models['svd'] = pickle.load(f)
    with open('models/birch_model.pkl', 'rb') as f:
        models['birch'] = pickle.load(f)
    
    # Load reduced data
    reduced_data = pd.read_pickle('data/processed/reduced_data.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None
    reduced_data = None

@app.route('/recommend', methods=['GET', 'POST'])  # Allow both GET and POST
def recommend():
    if request.method == 'POST':
        try:
            # Print request details for debugging
            print("Request received:")
            print("Headers:", dict(request.headers))
            print("Data:", request.get_json())

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Get input text
            title = data.get('title', '')
            subjects = data.get('subjects', '')
            synopsis = data.get('synopsis', '')
            
            # Print input data for debugging
            print(f"Processing request for title: {title}")
            
            # Combine text
            input_text = f"{title} {subjects} {synopsis}"
            
            # Transform input
            tfidf_features = models['tfidf'].transform([input_text])
            scaled_features = models['scaler'].transform(tfidf_features)
            reduced_features = models['svd'].transform(scaled_features)
            
            # Calculate similarities
            similarities = cosine_similarity(reduced_features, reduced_data)
            
            # Get top 5 similar indices
            top_indices = similarities[0].argsort()[-5:][::-1]
            top_scores = similarities[0][top_indices]
            
            # Load original data
            df = pd.read_csv('data/isbndb-caribbean-books.csv', encoding= 'latin1')
            df.fillna("Unknown", inplace=True)

            # Format results
            results = {
                'recommendations': [
                    {
                        'index': int(idx),
                        'similarity_score': float(score),
                        'title': df.iloc[idx]['title'] if pd.notna(df.iloc[idx]['title']) else "Unknown",
                        'subjects': df.iloc[idx]['subjects']  if pd.notna(df.iloc[idx]['subjects']) else "Unknown",
                        'publisher': df.iloc[idx]['publisher']  if pd.notna(df.iloc[idx].get('publisher', '')) else "Unknown",
                    }
                    for idx, score in zip(top_indices, top_scores)
                ]
            }
            
            print(f'RESULTS: {results}')

            return jsonify(results)
            
        except Exception as e:
            print(f"Error processing request: {e}")  # Debug print
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/')
def home():
    print("Book Recommendation API is running!")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
