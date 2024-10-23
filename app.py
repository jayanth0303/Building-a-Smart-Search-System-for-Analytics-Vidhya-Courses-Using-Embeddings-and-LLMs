import pandas as pd
data = pd.read_csv('AnalyticsVidhya.csv')

data.head(5)

# Check for missing values in each column
missingvalues = data.isnull().sum()

# Display rows with missing values
missingrows = data[data.isnull().any(axis=1)]
missingrows

# Drop rows where 'Time(Hours)' or 'Level' 
data.dropna(subset=['Level', 'Time(Hours)'], inplace=True)

import re
def cleantext(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
data['CourseTitle'] = data['CourseTitle'].apply(cleantext)
data['Description'] = data['Description'].apply(cleantext)

#Import Pre-trained Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


# Generate embeddings for the course descriptions
data['DescriptionEmbeddings'] = data['Description'].apply(lambda desc: model.encode(desc))
# Generate embeddings for course titles 
data['TitleEmbeddings'] = data['CourseTitle'].apply(lambda x: model.encode(x))


data.to_csv('courseswithembeddings.csv', index=False)

#Implement the Search Functionality
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set display options
pd.set_option('display.max_colwidth', 100) 
pd.set_option('display.width', 1000) 

def search_courses(query, data, model):
    # Generate embedding for the user query
    queryembedding = model.encode(query)
    
    # Compute cosine similarity between query and course descriptions
    similarities = cosine_similarity([queryembedding], list(data['DescriptionEmbeddings']))[0]
    
    # Top 5 results
    top5 = 5  
    topindices = similarities.argsort()[-top5:][::-1]
    
    # Retrieve the top results
    results = data.iloc[topindices]
    return results[['CourseTitle', 'Level', 'Category']]


#Create the User Interface with Gradio
import gradio as gr
data['CourseTitle'] = data['CourseTitle'].str.title()
def searchinterface(query):
    results = search_courses(query, data, model)
    return results.to_string(index=False)

# Create Gradio interface
interface = gr.Interface(
    fn=searchinterface, 
    inputs="text", 
    outputs="text", 
    title="Smart Engine for Free Courses",
    description="Enter Topic to find relevant free courses."
)

# Launch the interface
interface.launch()


