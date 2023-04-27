import pandas as pd
from sklearn.cluster import KMeans
from transformers import DistilBertTokenizer, DistilBertModel

import torch



data=pd.read_csv('quora_dataset/quora_duplicate_questions.tsv', sep='\t')
    
def return_all_sentences_in_dict():
    all_questions=dict()
    for index,row in data.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            id1=row["qid1"]
            id2=row["qid2"]
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            if sent1=="" and sent2=="":
                continue
            
            all_questions[id1]=row["question1"]
            all_questions[id2]=row["question2"]
    return all_questions

all_questions=return_all_sentences_in_dict()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define a function to get sentence embeddings
def get_sentence_embedding(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0].squeeze()
    return last_hidden_states.detach().numpy()

print("Getting all embeddings ")
# Get sentence embeddings
embeddings=dict()
for key,value in all_questions.items():
    if key not in embeddings:
        embeddings[key]=get_sentence_embedding(value)
print("Starting K means")
# Cluster the embeddings using KMeans
kmeans = KMeans(n_clusters=16, random_state=0).fit(embeddings)
print("K means done")
# Print the predicted labels for each sentence
count=0
categories1=[]
categories2=[]
is_same_category=[]
data=pd.read_csv('quora_dataset/quora_duplicate_questions.tsv', sep='\t')
    
for index,row in data.iterrows():
    if isinstance(row["question1"],str) and isinstance(row["question2"],str):
        id1=row["qid1"]
        id2=row["qid2"]
        sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
        sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
        if sent1=="" and sent2=="":
            categories1.append(0)
            categories2.append(0)
            is_same_category.append(0)
            continue
        label_1=kmeans.labels_[embeddings[id1]]
        label_2=kmeans.labels_[embeddings[id2]]
        categories1.append(label_1)
        categories2.append(label_2)

        if label_1 == label_2:
            is_same_category.append(1)
        else:
            is_same_category.append(0)

data["categories1"]=categories1
data["categories2"]=categories2

data["is_same_category"]==is_same_category

data.to_csv("quora_dataset/quora_duplicate_questions_with_categories.tsv",sep="\t",index=False)