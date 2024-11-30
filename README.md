# CariCon: **CariConnect** <sub>with Break Through Tech LA and Caribbean Literary Conference (CARICON)</sub>
Ranked recommendation system based on integrated clustering and LLM models. Matches upcoming authors with prospective publishers using input book information by gathering similar books from existing dataset through clustering and cosine similarity, while the LLM provides re-ranking and a generated explanation. <br />

## Methodology:
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

## Credits and Acknowledgments
Suggested Content: List of team members, advisors, supporters; also libraries or third-party services used in the project.
