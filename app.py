# -*- coding: utf-8 -*-
"""
@author: HP
"""

#Library imports
import uvicorn
from fastapi import FastAPI
from Sentences import sentences
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

#Create the FastAPI object
app = FastAPI()

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Index route,on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message':'Hello'}

@app.post('/Predict')
def Similarity_Score(data:sentences):
    data = data.dict()
    text1 = data['text1']
    text2 = data['text2']

    embedding_1 = model.encode(text1, convert_to_tensor=True)
    embedding_2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    simscore = float("{:.2f}".format(score))
    return {
        'similarity score': simscore
    }

#Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload

