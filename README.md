# CariCon: **CariConnect**
## with Break Through Tech LA and Caribbean Literary Conference (CARICON)
Ranked recommendation system based on integrated clustering and LLM models. Matches upcoming authors with prospective publishers using input book information by gathering similar books from existing dataset through clustering and cosine similarity, while the LLM provides re-ranking and a generated explanation. <br />

## Methods:
We utilized text data in csv's, translating and preprocessing the text data so that it was cleaned. This included dropping independent and missing publishers. We then vectorized the test data using BERT word embedding through HuggingFace Transformers.
The final model we selected was a Spectral clustering model from SciKit Learn. The number of clusters was selected based on highest silhouette score. We averaged around a number of clusters from 50 to 60, with a silhouette score from .45 to .50. Within the cluster, the cosine similarity of the datapoints with the input were calculated and ranked. We also integrated Retrieval Augmented Generation (RAG), where Langchain with an LLM was used to fetch additional matches. Both the clustering cosine similarity and RAG results were combined to be reranked by the LLAMA 3 70B LLM model, accessed through the Groq API.
For the purpose of the web application, we integrated our final clustering results by appending cluster assignments to the data. The website recieves an input, and the application preproccesses the input and matches it into a cluster, to which RAG fetches additional matches and reranks the conbined results with an explanation.


## Installation User Guide
APIs inclusde Groq and HuggingFace. The user must generate their own API keys and add them to a .env file. The API documentations are listed below.
HuggingFace provides access to LLMs and BERT transformers. Groq provides access to the LLAMA LLM, though LLAMA may be run locally instead.
The requirements are provided in the requirements.txt file. These requirements can be installed in the command line using the command pip install -r requirements.txt
The website uses Flask. To run the website, the Flask address may need be be adjusted. The port used is port 8000.
To use this application, run app.py. This will host the Flask website locally for access to the model and testing.
The notebook Generate_Spectral_Clustering_Model_Files.ipynb may be used to retrain the model and generate new files. This is only necessary if the model is to be trained on new data. Otherwise, the application has already been integrated with the necessary files.
Test inputs have been provided in test.csv. Testing can be done through the website interface with a variety of inputs.

## License

## Acknowledgments
Team:
TA:
Advisors:

APIs: HuggingFace, Groq
Libaries:
contractions==0.1.73
deep-translator==1.11.4
Flask==2.3.2
Flask-Cors==5.0.0
groq==0.12.0
langdetect==1.0.9
nltk==3.8.1
deep-translator==1.11.4
langdetect==1.0.9
numpy==1.26.4
pandas==2.2.3
python-dotenv
scikit-learn==1.5.2
chromadb==0.5.20
langchain==0.3.8
langchain_community==0.3.8
tiktoken==0.8.0
sentence_transformers==3.3.1
faiss-cpu==1.7.4
fpdf==1.7.2
werkzeug==2.3.3
matplotlib
