import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')

data=pd.read_csv('quora_dataset/quora_duplicate_questions_preprocessed.tsv', sep='\t')

def pre_process(data):
    all_questions=[]
    for index,row in data.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            id1=row["qid1"]
            id2=row["qid2"]
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            if sent1=="" and sent2=="":
                continue
            
            all_questions.append(row["question1"])
            all_questions.append(row["question2"])
    return all_questions

all_sentences=pre_process(data)

all_sentences_embeddings = model.encode(all_sentences)
print(all_sentences_embeddings.shape)

# Save embeddings to file
np.save('all_sentence_embeddings.npy', all_sentences_embeddings)

