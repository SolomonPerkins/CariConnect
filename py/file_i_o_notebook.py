# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle

tfidf = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))
max_scaler = pickle.load(open('../models/maxabs_scaler.pkl', 'rb'))
svd = pickle.load(open('../models/svd_model.pkl', 'rb'))
cmodel = pickle.load(open('../models/birch_model.pkl', 'rb'))
df_match = pd.read_csv('../data/df_matching.csv', index_col ='Unnamed: 0')
df_reduced = pd.read_csv('../data/df_reduced.csv', index_col ='Unnamed: 0')

# Text Preprocessing
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import contractions
def preprocess(s):
    s=str(s)
    s = s.lower() # lowercase
    s = contractions.fix(s) # expand contractions
    s = re.sub(r'\\n', ' ', s) # remove \n
    s = re.sub(r'http\S+', '', s) # remove url
    s = re.sub(r'<.*?>', '', s) # remove html
    s = re.sub(r'\d+', '', s) # remove numbers
    s = re.sub(r'[^\w\s]', ' ', s) # remove punctuation and special characters
    s = word_tokenize(s) # tokenize
    s = [w for w in s if w not in set(stopwords.words('english'))] # stop words
    s = [stemmer.stem(w) for w in s] # stemming
    return " ".join(s) # white spaces
def preprocess_transform(s):
    preprocessed = preprocess(s)
    transformed = tfidf.transform([preprocessed]).toarray()
    return transformed

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0

# Input language determinate
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
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except:
        return text
detector = LanguageDetector()

# column weights
title_weight=.05
subject_weight=.8
synopsis_weight=.15

# Clean and vectorize input
title_input = "Anansi"
subject_input = "Akan Folklore"
synopsis_input = "Anansi or Ananse (/əˈnɑːnsi/ ə-NAHN-see; literally translates to spider) is an Akan folktale character associated with stories, wisdom, knowledge, and trickery, most commonly depicted as a spider, in Akan folklore.[1] Taking the role of a trickster, he is also one of the most important characters of West African, African American and West Indian folklore. Originating in Ghana, these spider tales were transmitted to the Caribbean by way of the transatlantic slave trade.[2]  Anansi is best known for his ability to outsmart and triumph over more powerful opponents through his use of cunning, creativity and wit.[3] Despite taking on a trickster role, Anansi often takes centre stage in stories and is commonly portrayed as both the protagonist and antagonist."
detector.set_text(synopsis_input)
detect_input = detector.detect_language()
synopsis_input = translate_to_english(synopsis_input)
lang_input = detect_input # detect language, or selected language

clean_title_input = title_weight*preprocess_transform(title_input)
clean_subject_input = subject_weight*preprocess_transform(subject_input)
clean_synopsis_input = synopsis_weight*preprocess_transform(synopsis_input)
input_df = pd.DataFrame(np.concatenate([clean_title_input, clean_subject_input, clean_synopsis_input], axis=1))

# scale input
input_df = max_scaler.transform(input_df)
# reduce dimensions of input
svd_input = svd.transform(input_df)
red_input = svd_input

# predict test input
input_pred = cmodel.predict(red_input)[0]

# vector subset where label matches
subset = df_match[df_match['label'] == input_pred]
sim_df = df_reduced.copy()
sim_subset = sim_df.loc[subset.index]

# calculate cosine similarity within that cluster
from sklearn.metrics.pairwise import cosine_similarity
simscore = cosine_similarity
similarity = simscore(red_input, sim_subset).T #input

# append similarity to dataframe
subset['similarity'] = similarity
subset.loc[df_match['language'] == lang_input, 'similarity']+=1 # prioritize same language

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
    title = df_match.loc[index]['title']
    subjects = df_match.loc[index]['subjects']
    synopsis = df_match.loc[index]['synopsis']
    publisher = df_match.loc[index]['publisher']
    language = df_match.loc[index]['language']
    return title, subjects, synopsis, publisher, language

for i in range(len(top_index)):
    top_titles.append(book_info(top_index[i])[0])
    top_subjects.append(book_info(top_index[i])[1])
    top_synopsis.append(book_info(top_index[i])[2])
    top_publishers.append(book_info(top_index[i])[3])
    top_languages.append(book_info(top_index[i])[4])

from groq import Groq
client = Groq(
    #api_key="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",
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
            "content": f"You are a knowledgable and straightforward assistant with experience in literature and literary analysis, and knowledge of the publishing industry. You are explaining to authors which books are most similar to theirs using criteria such as themes, plot, character, setting, and tone. Your responses are concise and academic, strictly providing lists and explanations. Structure the response in a list, where each entry is formatted, with elaboration as necessary. (Do not mention the books' alphabetical labels A B C D E) : \n (Rank number). (Publisher) \n Published in (Language)\n Explanation: (Publisher) published (Title), a (Subject) book that is similar to your book because (Explanation, 3 sentence justification of the book similarities). Therefore, your book appeals to (Publisher)'s sector of the market."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": f"Using the input book information as a reference, compare and rank the similarity of books A, B, C, D, and E to the reference. Use literary analysis to evaluate each book based on thematic alignment, such as central conflict, character focus, and setting, and stylistic features such as narrative tone and diction. \n Provide a list of the respective publishers of the books in descending order of book similarity to the reference book with a 3 sentence justification of the book similarities and ranking, citing specific aspects of the book information and synopsis that supports your assessment. Structure the response in a list, where each entry is formatted, with elaboration as necessary. (Do not mention the books' alphabetical labels A B C D E) : \n (Rank number). (Publisher) \n Published in (Language)\n Explanation: (Publisher) published (Title), a (Subject) book that is similar to your book because (Explanation, 3 sentence justification of the book similarities). Therefore, your book appeals to (Publisher)'s sector of the market. \n \n Book Reference: \n Title: {title_input}. \n Subject: {subject_input} \n Synopsis: {synopsis_input} \n \n Book A: \n Title: {top_titles[0]} \n Subject: {top_subjects[0]} \n Synopsis: {top_synopsis[0]} \n Publisher: {top_publishers[0]} \n Language: {top_languages[0]} \n \n Book B: \n Title: {top_titles[1]} \n Subject: {top_subjects[1]} \n Synopsis: {top_synopsis[1]} \n Publisher: {top_publishers[1]} \n Language: {top_languages[1]} \n \n Book C: \n Title: {top_titles[2]} \n Subject: {top_subjects[2]} \n Synopsis: {top_synopsis[2]} \n Publisher: {top_publishers[2]} \n Language: {top_languages[2]} \n \n Book D: \n Title: {top_titles[3]} \n Subject: {top_subjects[3]} \n Synopsis: {top_synopsis[3]} \n Publisher: {top_publishers[3]} \n Language: {top_languages[3]} \n \n Book E: \n Title: {top_titles[4]} \n Subject: {top_subjects[4]} \n Synopsis: {top_synopsis[4]} \n Publisher: {top_publishers[4]} \n Language: {top_languages[4]}"

        }

    ],

    # The language model which will generate the completion.
    model= "llama3-70b-8192",#"llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"

    #
    # Optional parameters
    #

    # Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=0.3,

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
output = GoogleTranslator(source='auto', target=lang_input).translate(output)
print(output)