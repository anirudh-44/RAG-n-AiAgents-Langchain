from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#print(doc_embeddings)
#print(query_embedding)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
#print(scores)
#print(scores[0])

#print(list(enumerate(scores)))

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
''' sorting based on the similarity score in asc order and retreiving the highest
    score which is the last one
'''

print(query)
print(documents[index])
print("similarity score is:", score)