from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import re

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


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

from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Load models at startup
try:
    with open('models/spectral_model.pkl', 'rb') as f:
        spectral = pickle.load(f)

    # Load reduced data
    df_model = pd.read_csv('data/df_spec_modeling.csv')
    weighted_embeddings = np.load('data/weighted_embeddings.npy')
    similarity_matrix = np.load('data/similarity_matrix.npy')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    spectral = None
    df_model = None
    weighted_embeddings = None
    similarity_matrix = None


model = SentenceTransformer('all-MiniLM-L6-v2')  # paraphrase- ... perform less good
llm = HuggingFaceHub(repo_id="microsoft/Phi-3-mini-4k-instruct", model_kwargs={"temperature": 0.6})

embeddings = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
# embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')
docsearch = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)


stemmer = SnowballStemmer("english")

def preprocess(s):
    s=str(s)
    s = s.lower() # lowercase !
    s = contractions.fix(s) # expand contractions
    s = re.sub(r'\n', ' ', s) # remove \n
    s = re.sub(r'http\S+', '', s) # remove url
    s = re.sub(r'<.*?>', '', s) # remove html
    s = re.sub(r'\d+', '', s) # remove numbers
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^\w\s]', '', s) # remove punctuation and special characters
    s = word_tokenize(s) # tokenize
    s = [w for w in s if w not in set(stopwords.words('english'))] # stop words
    s = [stemmer.stem(w) for w in s] # stemming
    return " ".join(s) # white spaces

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

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
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

            # Get input
            title_input = data.get('title', '')
            subject_input = data.get('subjects', '')
            synopsis_input = data.get('synopsis', '')
            # Print input data for debugging
            print(f"Processing request for title: {title_input}")
            title_input = preprocess(title_input)
            synopsis_input = preprocess(synopsis_input)
            detector.set_text(synopsis_input)
            synopsis_input = translate_to_english(synopsis_input)
            subj_prompt = f"Generate a brief list of subjects and categories based on this synopsis: {synopsis_input}. Follow the format structure: 1. Category: <category> 2. Category: <category> 3. Category <category> 4. EndOutput"
            if subject_input == "":
                subject_input = llm(subj_prompt)
                subject_input = subject_input.split("EndOutput")[1]
                subject_input = re.findall(r'Category: (.+)', subject_input)
            subject_input = preprocess(subject_input)
            subject_input = translate_to_english(subject_input)

            print(synopsis_input)

            # 1.Generate weighted embedding for new input
            new_subject_embedding = model.encode(subject_input, batch_size=1)
            new_synopsis_embedding = model.encode(synopsis_input, batch_size=1) if synopsis_input else np.zeros_like(
                new_subject_embedding)

            if new_synopsis_embedding.shape[0] == 0:
                synopsis_weight = 1
                subject_weight = 0
            else:
                subject_weight = 0.9
                synopsis_weight = 0.1

            new_weighted_embedding = (
                    subject_weight * new_subject_embedding +
                    synopsis_weight * new_synopsis_embedding
            )

            # 2.Calculate cluster centroids from original similarity matrix
            cluster_centroids = {}
            for cluster_num in np.unique(df_model['cluster']):  # 'clusters' is the output of Spectral Clustering
                cluster_indices = np.where(df_model['cluster'] == cluster_num)[0]
                cluster_centroids[cluster_num] = np.mean(weighted_embeddings[cluster_indices], axis=0)

            # 3.Compute similarity with new data point
            similarities = {}
            for cluster_num, centroid in cluster_centroids.items():
                similarity = cosine_similarity(new_weighted_embedding.reshape(1, -1), centroid.reshape(1, -1))[0][0]
                similarities[cluster_num] = similarity

            # 4.Assign to the most similar cluster
            assigned_cluster = max(similarities, key=similarities.get)

            print(f"Assigned Cluster: {assigned_cluster}")
            print(f"Similarity Scores: {similarities}")

            cluster_books_indices = df_model[df_model['cluster'] == assigned_cluster].index

            similarity_scores = cosine_similarity(new_weighted_embedding.reshape(1, -1), weighted_embeddings[cluster_books_indices]).T

            df_cluster = df_model[df_model['cluster'] == assigned_cluster]
            df_cluster['similarity'] = similarity_scores
            similar_books = df_cluster.sort_values(by='similarity', ascending=False, inplace=False)
            top_books = similar_books.head(5) if len(similar_books) >= 5 else similar_books
            top_books_details = df_model.loc[top_books.index]
            top_cluster_index = top_books_details.index.tolist()
            print(top_cluster_index)

            retriever = docsearch.as_retriever()

            # Define the prompt
            prompt_template = """
            Compare the book given in the question with others in the retriever. Focus primarily on genre (subject) as the main matching criterion, followed by details in the title and synopsis.

            Return the top 5 most similar books. For each book, include:
            - The original title.
            - The reason for similarity (based on genre, title, and synopsis).
            - The index or identifier of the book (if available).

            question: {question}
            context: {context}

            Provide your response in this structured format:
            1. Index: <index>, Title: <title>, Reason: <reason>
            2. Index: <index>, Title: <title>, Reason: <reason>
            3. Index: <index>, Title: <title>, Reason: <reason>
            4. Index: <index>, Title: <title>, Reason: <reason>
            5. Index: <index>, Title: <title>, Reason: <reason>
            """

            # Create the prompt object
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            # Create the RetrievalQA object
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )

            # Query
            query = f"Find the top 5 books most similar to one with subject '{subject_input}', title '{title_input}', and synopsis '{synopsis_input}'. Focus on genre and subject as primary matching criteria."

            # Run the QA and get results
            results = qa.run({"query": query})
            print(results)

            titles = re.findall(r'title: (.+)', results)

            # Step 2: Match extracted titles to the title column in df_model and get indices
            matched_indices = []
            for title in titles[:5]:  # Only process the first 5 titles
                match = df_model[df_model["title"] == title]
                if not match.empty:
                    matched_indices.append(match.index[0])  # Get the corresponding index from df_model

            # Step 3: Output the results
            print("Extracted Titles:", titles[:5])
            print("Corresponding Indices:", matched_indices)
            top_RAG_index = matched_indices

            top_index = top_cluster_index + top_RAG_index
            top_index = list(set(top_index))

            top_titles = []
            top_subjects = []
            top_synopsis = []
            top_publishers = []

            def book_info(index):
                title = df_model.loc[index]['title']
                subjects = df_model.loc[index]['subjects']
                synopsis = df_model.loc[index]['synopsis']
                publisher = df_model.loc[index]['publisher']
                return title, subjects, synopsis, publisher

            for i in range(len(top_index)):
                top_titles.append(book_info(top_index[i])[0])
                top_subjects.append(book_info(top_index[i])[1])
                top_synopsis.append(book_info(top_index[i])[2])
                top_publishers.append(book_info(top_index[i])[3])
                print(book_info(top_index[i])[0])  # titles

            # Commented out IPython magic to ensure Python compatibility.
            # %env GROQ_API_KEY=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

            user_content = f"""
            Using the input book information as a reference, compare and rank the similarity of books to the reference. Use literary analysis to evaluate each book based on thematic alignment, such as central conflict, character focus, and setting, and stylistic features such as narrative tone and diction.

            Provide a list of the respective publishers of the books in descending order of book similarity to the reference book with a 3-sentence justification of the book similarities and ranking, citing specific aspects of the book information and synopsis that supports your assessment.

            Structure the response in a list, where each entry is formatted, with elaboration as necessary:
                (Rank number). (Publisher)
                Explanation: (Publisher) published (Title), a (Subject) book that is similar to your book because (Explanation, 3-sentence justification of the book similarities). Therefore, your book appeals to (Publisher)'s sector of the market.

            Book Reference:
            Title: {title_input}.
            Subject: {subject_input}.
            Synopsis: {synopsis_input}.
            """

            # Add dynamically the list of books
            for i in range(len(top_titles)):
                user_content += f"\n\nBook {chr(65 + i)}:\n"
                user_content += f"Title: {top_titles[i]}\n"
                user_content += f"Subject: {top_subjects[i]}\n"
                user_content += f"Synopsis: {top_synopsis[i]}\n"
                user_content += f"Publisher: {top_publishers[i]}"

            print(user_content)  # For debugging or review

            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )

            chat_completion = client.chat.completions.create(
                #
                # Required parameters
                #
                messages=[
                    # Set an optional system message. This sets the behavior of the
                    # assistant and can be used to provide specific instructions for
                    # how it should behave throughout the conversation.
                    {
                        "role": "system",
                        "content": "You are a knowledgable and straightforward assistant with experience in literature and literary analysis, and knowledge of the publishing industry. You are explaining to authors which books are most similar to theirs using criteria such as themes, plot, character, setting, and tone. Your responses are concise and academic, strictly providing lists and explanations."
                    },
                    # Set a user message for the assistant to respond to.
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],

                # The language model which will generate the completion.
                model="llama3-70b-8192",  # "llama3-8b-8192", #llama3-70b-8192

                #
                # Optional parameters
                #

                # Controls randomness: lowering results in less random completions.
                # As the temperature approaches zero, the model will become deterministic
                # and repetitive.
                temperature=0.5,

                # The maximum number of tokens to generate. Requests can use up to
                # 32,768 tokens shared between prompt and completion.
                max_tokens=8000,

                # Controls diversity via nucleus sampling: 0.5 means half of all
                # likelihood-weighted options are considered.
                top_p=1,

                # A stop sequence is a predefined or user-specified text string that
                # signals an AI to stop generating content, ensuring its responses
                # remain focused and concise. Examples include punctuation marks and
                # markers like "[end]".
                stop=None,

                # If set, partial message deltas will be sent.
                stream=False,
            )
            # Print the completion returned by the LLM.
            output = chat_completion.choices[0].message.content
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