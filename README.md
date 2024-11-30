# CariCon: **CariConnect**
## with Break Through Tech LA and Caribbean Literary Conference (CARICON)
Ranked recommendation system based on integrated clustering and LLM models. Matches upcoming authors with prospective publishers using input book information by gathering similar books from existing dataset through clustering and cosine similarity, while the LLM provides re-ranking and a generated explanation.

## Methods:
We utilized text data in csv's, translating and preprocessing the text data so that it was cleaned. This included dropping independent and missing publishers. We then vectorized the test data using BERT word embedding through HuggingFace Transformers.<br />
The final model we selected was a Spectral clustering model from SciKit Learn. The number of clusters was selected based on highest silhouette score. We averaged around a number of clusters from 50 to 60, with a silhouette score from .45 to .50.<br />
Within the cluster, the cosine similarity of the datapoints with the input were calculated and ranked.<br />
We also integrated Retrieval Augmented Generation (RAG), where Langchain with an LLM was used to fetch additional matches.<br />
Both the clustering cosine similarity and RAG results were combined to be reranked by the LLAMA 3 70B LLM model, accessed through the Groq API.<br />
For the purpose of the web application, we integrated our final clustering results by appending cluster assignments to the data. The website recieves an input, and the application preproccesses the input and matches it into a cluster, to which RAG fetches additional matches and reranks the conbined results with an explanation.

## Installation User Guide
APIs inclusde Groq and HuggingFace. The user must generate their own API keys and add them to a .env file. The API documentations are listed below.<br />
[HuggingFace](https://huggingface.co/docs/hub/en/security-tokens)<br />
[Groq](https://console.groq.com/keys)<br />
HuggingFace provides access to LLMs and BERT transformers. Groq provides access to the LLAMA LLM, though LLAMA may be run locally instead.<br />
The requirements are provided in the requirements.txt file. These requirements can be installed in the command line using the command pip install -r requirements.txt<br />
The website uses Flask. To run the website, the Flask address may need be be adjusted. The port used is port 8000.<br />
To use this application, run app.py. This will host the Flask website locally for access to the model and testing.<br />
The notebook Generate_Spectral_Clustering_Model_Files.ipynb may be used to retrain the model and generate new files, inclusing silhouette score visual analysis. This is only necessary if the model is to be trained on new data. Otherwise, the application has already been integrated with the necessary files.<br />
Test inputs have been provided in test.csv. Testing can be done through the website interface with a variety of inputs.

## Acknowledgments
Team: Ashley Camacho-Medellin, Khant Nyi Hlaing, [Kaylin (Kienn) Nguyen](https://github.com/kn21), Eileen (Yiming) Xue, Grace (Yunjin) Zhu <br />
TAs: Arjun Aggarwal, Rebecca Aurelia Dsouza <br />
Advisors: V. Steve Russell, Solomon Perkins<br />

APIs: HuggingFace, Groq<br />
Libaries: 
chromadb,
contractions,
deep-translator,
faiss-cpu,
Flask,
Flask-Cors,
groq,
langdetect,
matplotlib,
nltk,
numpy,
pandas,
python-dotenv,
scikit-learn,
langchain,
langchain_community,
sentence_transformers,
tiktoken,
werkzeug
