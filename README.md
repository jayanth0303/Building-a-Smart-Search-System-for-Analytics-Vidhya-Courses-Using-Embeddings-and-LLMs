Project Title: Building-a-Smart-Search-System-for-Analytics-Vidhya-Courses-Using-Embeddings-and-LLMs
Project Description:

This project involves building an intelligent, keyword-based search tool that allows users to efficiently discover free courses on the Analytics Vidhya platform. The system utilizes natural language processing (NLP) and semantic search techniques to enhance traditional keyword search with contextual understanding.

The project follows these key phases:

Data Collection: Scraped free course information (titles, descriptions, categories, levels, etc.) from the Analytics Vidhya website.

Data Preprocessing: Cleaned and normalized the textual data by handling missing values, converting text to lowercase, and removing special characters.

Embedding Generation: Used a pre-trained Sentence-BERT model (all-MiniLM-L6-v2) to convert course titles and descriptions into dense vector embeddings, capturing semantic meaning.

Search Engine Development: Implemented a semantic search function that compares user queries against course embeddings using cosine similarity to retrieve the top 5 most relevant courses.

Interface Creation: Built an interactive web-based interface using Gradio, enabling users to input queries and view results with ease.

Deployment (Optional): Prepared the tool for deployment on Hugging Face Spaces to make it publicly accessible.

Tech Stack:
Python (Pandas, NumPy, Scikit-learn)
NLP: Sentence Transformers (BERT-based embeddings)
Interface: Gradio
Optional Deployment: Hugging Face Spaces

Key Features:
Semantic search for high accuracy and relevance
Real-time results for user queries
Clean and minimal user interface
Scalable embedding-based architecture

This smart search engine demonstrates how NLP and machine learning models can transform static datasets into interactive, intelligent tools for education discovery.
