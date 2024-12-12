# CariConnect: Ranked Recommendation System
CariConnect is a recommendation system that matches upcoming authors with prospective publishers using a combination of clustering, cosine similarity, and Retrieval-Augmented Generation (RAG) techniques. This project integrates advanced machine learning models and LLMs for personalized recommendations and detailed explanations.

## Features
- Spectral Clustering: Groups books into clusters using BERT embeddings.
- Cosine Similarity: Ranks similar books within clusters.
- Retrieval-Augmented Generation (RAG): Enhances matches and provides explanations using LangChain and LLAMA LLM.
- An interactive web interface for user input and results.

## Prerequisites

- Obtain API keys for HuggingFace and Groq.
- Add keys to a .env file in the project root.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/khantnhl/cari-connect.git
cd cari-connect
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application:

```bash
run app.py
#Access at: http://127.0.0.1:8000
```


### 4. Optional: To retrain the Spectral Clustering model, use the provided notebook:
```bash
jupyter notebook Generate_Spectral_Clustering_Model_Files.ipynb
```

Team: Ashley Camacho-Medellin, Khant Nyi Hlaing, Kaylin (Kienn) Nguyen, Eileen (Yiming) Xue, Grace (Yunjin) Zhu
TAs: Arjun Aggarwal, Rebecca Aurelia Dsouza
Advisors: V. Steve Russell, Solomon Perkins

License
This project is licensed under the MIT License. 
