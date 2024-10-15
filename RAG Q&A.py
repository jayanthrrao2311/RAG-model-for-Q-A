!pip install openai pinecone-client transformers 

import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance
pc = Pinecone(
    api_key="xxxxxxx-xxxxx-xxxxx-xxxxx-xxxxxxxxxx"  # Replace with your Pinecone API key
)

# Create Pinecone index if it doesn't exist
index_name = "qa-bot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Example for GPT-3.5 embeddings
        metric='cosine',  # Or another metric like 'euclidian'
        spec=ServerlessSpec(
            cloud='aws',  # Choose the cloud provider, e.g., 'gcp' or 'aws'
            region='us-east-1'  # Choose the region, e.g., 'us-east1-gcp', 'us-west-2'
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Now, the `index` object can be used for inserting/querying vectors

import openai

openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" Replace with your openai API key

def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

def query_embedding(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Average pooling for simplicity

def retrieve_similar_docs(query):
    vector = query_embedding(query)  # Get the embedding for the query
    response = index.query(vector=vector.tolist(), top_k=5, include_metadata=True)  # Use keyword arguments and convert to list
    return response['matches']  

def generate_augmented_answer(query):
    docs = retrieve_similar_docs(query)
    context = " ".join([doc["metadata"]["text"] for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    return generate_answer(prompt)

# Example usage
query = "What are the benefits of buying shares?"
answer = generate_augmented_answer(query)
print(answer)
