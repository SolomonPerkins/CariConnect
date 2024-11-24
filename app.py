from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
import contractions

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator

from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

load_dotenv()

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
    match_data = pd.read_csv('data/df_matching.csv', index_col='Unnamed: 0')
    reduced_data = pd.read_csv('data/df_reduced.csv', index_col = 'Unnamed: 0')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None
    match_data = None
    reduced_data = None

stemmer = SnowballStemmer("english")
def preprocess(s):
    s=str(s)
    s = s.lower() # lowercase
    s = contractions.fix(s) # expand contractions
    s = str.replace(r'\\n', ' ', s) # remove \n
    s = str.replace(r'http\S+', '', s) # remove url
    s = str.replace(r'<.*?>', '', s) # remove html
    s = str.replace(r'\d+', '', s) # remove numbers
    s = str.replace(r'[^\w\s]', ' ', s) # remove punctuation and special characters
    s = word_tokenize(s) # tokenize
    s = [w for w in s if w not in set(stopwords.words('english'))] # stop words
    s = [stemmer.stem(w) for w in s] # stemming
    return " ".join(s) # white spaces
def preprocess_transform(s):
    preprocessed = preprocess(s)
    transformed = models['tfidf'].transform([preprocessed]).toarray()
    return transformed

# Input language determinate
DetectorFactory.seed = 0
class LanguageDetector:
    def __init__(self, text=""):
        self.text = text

    def set_text(self, text):
        self.text = text

    def detect_language(self):
        try:
            language_code = detect(self.text)
            return language_code
        except LangDetectException:
            language_code = "en"
            return language_code
def translate_to_english(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    return translated

detector = LanguageDetector()

# Define translations for each language
translations = {
    'English': {
        'title': 'Submit Book Details',
        'book_name': 'Name of the Book',
        'language': 'Language',
        'category': 'Category',
        'synopsis': 'Synopsis (Optional)',
        'submit': 'Submit',
        'feedback_title': 'Book Submission Result',
        'name_label': 'Book Name:',
        'language_label': 'Language:',
        'category_label': 'Category:',
        'synopsis_label': 'Synopsis:',
        'publisher_label': 'Publisher:',
        'probability_label': 'Probability of Accepting:',
        'related_book_label': 'Related Book Published:',
        'submit_button': 'Submit another book'
    },
    'Dutch': {
        'title': 'Boekdetails indienen',
        'book_name': 'Naam van het boek',
        'language': 'Taal',
        'category': 'Categorie',
        'synopsis': 'Samenvatting (Optioneel)',
        'submit': 'Verzenden',
        'feedback_title': 'Resultaat van boekinzending',
        'name_label': 'Boeknaam:',
        'language_label': 'Taal:',
        'category_label': 'Categorie:',
        'synopsis_label': 'Samenvatting:',
        'publisher_label': 'Uitgever:',
        'probability_label': 'Aanvaardingskans:',
        'related_book_label': 'Gerelateerd boek gepubliceerd:',
        'submit_button': 'Nog een boek indienen'
    },
    'French': {
        'title': 'Soumettre les détails du livre',
        'book_name': 'Nom du livre',
        'language': 'Langue',
        'category': 'Catégorie',
        'synopsis': 'Résumé (Facultatif)',
        'submit': 'Soumettre',
        'feedback_title': 'Résultat de la soumission du livre',
        'name_label': 'Nom du livre:',
        'language_label': 'Langue:',
        'category_label': 'Catégorie:',
        'synopsis_label': 'Résumé:',
        'publisher_label': 'Éditeur:',
        'probability_label': 'Probabilité d\'acceptation:',
        'related_book_label': 'Livre publié similaire:',
        'submit_button': 'Soumettre un autre livre'
    },
    'Haitian Creole': {
        'title': 'Soumèt detay liv la',
        'book_name': 'Non liv la',
        'language': 'Lang',
        'category': 'Kategori',
        'synopsis': 'Rezime (Opsyonèl)',
        'submit': 'Soumèt',
        'feedback_title': 'Rezilta soumèt liv la',
        'name_label': 'Non liv la:',
        'language_label': 'Lang:',
        'category_label': 'Kategori:',
        'synopsis_label': 'Rezime:',
        'publisher_label': 'Piblikatè:',
        'probability_label': 'Pwobabilite Akseptasyon:',
        'related_book_label': 'Liv ki gen rapò pibliye:',
        'submit_button': 'Soumèt yon lòt liv'
    },
    'Papiamentu': {
        'title': 'Entregá detaye di e buki',
        'book_name': 'Nòm di e buki',
        'language': 'Idioma',
        'category': 'Kategoria',
        'synopsis': 'Resumenchi (Opshonal)',
        'submit': 'Entregá',
        'feedback_title': 'Resultado di entrega di e buki',
        'name_label': 'Nòm di e buki:',
        'language_label': 'Idioma:',
        'category_label': 'Kategoria:',
        'synopsis_label': 'Resumenchi:',
        'publisher_label': 'Editoria:',
        'probability_label': 'Probabilidat di akseptashon:',
        'related_book_label': 'Buki relashona publika:',
        'submit_button': 'Entregá otro buki'
    },
    'Spanish': {
        'title': 'Enviar detalles del libro',
        'book_name': 'Nombre del libro',
        'language': 'Idioma',
        'category': 'Categoría',
        'synopsis': 'Sinopsis (Opcional)',
        'submit': 'Enviar',
        'feedback_title': 'Resultado de la presentación del libro',
        'name_label': 'Nombre del libro:',
        'language_label': 'Idioma:',
        'category_label': 'Categoría:',
        'synopsis_label': 'Sinopsis:',
        'publisher_label': 'Editorial:',
        'probability_label': 'Probabilidad de aceptación:',
        'related_book_label': 'Libro relacionado publicado:',
        'submit_button': 'Enviar otro libro'
    }
}

# column weights
title_weight=.05
subject_weight=.8
synopsis_weight=.15

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
@app.route('/recommend', methods=['GET', 'POST'])  # Allow both GET and POST
def recommend():
    # set language
    selected_language = request.args.get('language', 'English')
    language_content = translations.get(selected_language, translations['English'])
    translation = translations[selected_language]

    if request.method == 'POST':
        try:
            # Print request details for debugging
            print("Request received:")
            print("Headers:", dict(request.headers))
            print("Data:", request.get_json())

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Get input
            title_input = data.get('title', '')
            subject_input = data.get('subjects', '')
            synopsis_input = data.get('synopsis', '')
            lang_input = data.get('language', '')  # Empty if not selected

            # Print input data for debugging
            print(f"Processing request for title: {title_input}")

            detector.set_text(synopsis_input)
            detect_input = detector.detect_language()
            synopsis_input = translate_to_english(synopsis_input)
            if not lang_input:
                lang_input = detect_input

            clean_title_input = title_weight * preprocess_transform(title_input)
            clean_subject_input = subject_weight * preprocess_transform(subject_input)
            clean_synopsis_input = synopsis_weight * preprocess_transform(synopsis_input)
            input_df = pd.DataFrame(
                np.concatenate([clean_title_input, clean_subject_input, clean_synopsis_input], axis=1))

            # scale input
            input_df = models['scaler'].transform(input_df)
            # reduce dimensions of input
            red_input = models['svd'].transform(input_df)

            # predict test input
            input_pred = models['birch'].predict(red_input)[0]

            # vector subset where label matches
            subset = match_data[match_data['label'] == input_pred]
            sim_subset = reduced_data.loc[subset.index]

            # calculate cosine similarity within that cluster
            simscore = cosine_similarity
            similarity = simscore(red_input, sim_subset).T  # input

            # append similarity to dataframe
            subset['similarity'] = similarity
            subset.loc[match_data['language'] == lang_input, 'similarity'] += 1  # prioritize same language

            # top 5 similar
            top_subset = subset.sort_values(by='similarity', ascending=False).head(5)
            top_index = top_subset.index

            # get information of top 5
            top_titles = []
            top_subjects = []
            top_synopsis = []
            top_publishers = []
            top_languages = []

            def book_info(index):
                title = match_data.loc[index]['title']
                subjects = match_data.loc[index]['subjects']
                synopsis = match_data.loc[index]['synopsis']
                publisher = match_data.loc[index]['publisher']
                language = match_data.loc[index]['language']
                return title, subjects, synopsis, publisher, language

            for i in range(len(top_index)):
                top_titles.append(book_info(top_index[i])[0])
                top_subjects.append(book_info(top_index[i])[1])
                top_synopsis.append(book_info(top_index[i])[2])
                top_publishers.append(book_info(top_index[i])[3])
                top_languages.append(book_info(top_index[i])[4])

            print('calculations complete')
            client = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    # Set an optional system message.
                    {
                        "role": "system",
                        "content": "You are a knowledgable and straightforward assistant with experience in literature and literary analysis, and knowledge of the publishing industry. You are explaining to authors which books are most similar to theirs using criteria such as themes, plot, character, setting, and tone. Your responses are concise and academic, strictly providing lists and explanations. Structure the response in a list, where each entry is formatted, with elaboration as necessary. (Do not mention the books' alphabetical labels A B C D E) : \n (Rank number). (Publisher) \n Published in (Language)\n Explanation: (Publisher) published (Title), a (Subject) book that is similar to your book because (Explanation, 3 sentence justification of the book similarities). Therefore, your book appeals to (Publisher)'s sector of the market."
                    },
                    # Set a user message for the assistant to respond to.
                    {
                        "role": "user",
                        "content": f"Using the input book information as a reference, compare and rank the similarity of books A, B, C, D, and E to the reference. Use literary analysis to evaluate each book based on thematic alignment, such as central conflict, character focus, and setting, and stylistic features such as narrative tone and diction. \n Provide a list of the respective publishers of the books in descending order of book similarity to the reference book with a 3 sentence justification of the book similarities and ranking, citing specific aspects of the book information and synopsis that supports your assessment. Structure the response in a list, where each entry is formatted, with elaboration as necessary. (Do not mention the books' alphabetical labels A B C D E) : \n (Rank number). (Publisher) \n Published in (Language)\n Explanation: (Publisher) published (Title), a (Subject) book that is similar to your book because (Explanation, 3 sentence justification of the book similarities). Therefore, your book appeals to (Publisher)'s sector of the market. \n \n Book Reference: \n Title: {title_input}. \n Subject: {subject_input} \n Synopsis: {synopsis_input} \n \n Book A: \n Title: {top_titles[0]} \n Subject: {top_subjects[0]} \n Synopsis: {top_synopsis[0]} \n Publisher: {top_publishers[0]} \n Language: {top_languages[0]} \n \n Book B: \n Title: {top_titles[1]} \n Subject: {top_subjects[1]} \n Synopsis: {top_synopsis[1]} \n Publisher: {top_publishers[1]} \n Language: {top_languages[1]} \n \n Book C: \n Title: {top_titles[2]} \n Subject: {top_subjects[2]} \n Synopsis: {top_synopsis[2]} \n Publisher: {top_publishers[2]} \n Language: {top_languages[2]} \n \n Book D: \n Title: {top_titles[3]} \n Subject: {top_subjects[3]} \n Synopsis: {top_synopsis[3]} \n Publisher: {top_publishers[3]} \n Language: {top_languages[3]} \n \n Book E: \n Title: {top_titles[4]} \n Subject: {top_subjects[4]} \n Synopsis: {top_synopsis[4]} \n Publisher: {top_publishers[4]} \n Language: {top_languages[4]}"
                    }
                ],
                # The language model which will generate the completion.
                model="llama3-70b-8192",
                # Controls randomness: lowering results in less random completions.
                temperature=0.3,
                # The maximum number of tokens to generate.
                max_tokens=8000,
                # Controls diversity via nucleus sampling
                top_p=1,
                # A stop sequence is a predefined or user-specified text string
                stop=None,
                # If set, partial message deltas will be sent.
                stream=False,
            )

            # Print the completion returned by the LLM.
            output = chat_completion.choices[0].message.content

            print(output)
            #output = GoogleTranslator(source='auto', target=lang_input).translate(output)

            print(output)

            # Format results
            results = {
                'recommendations': [
                    {
                        'output': output,
                    }
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
