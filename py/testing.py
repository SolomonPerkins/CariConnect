from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
import contractions
import re

load_dotenv()


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

print("preprocessing")
print(preprocess("Things."))
print("endpreprocess")