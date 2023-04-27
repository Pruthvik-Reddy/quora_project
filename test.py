from transformers import BertTokenizer
from torch.utils.data import DataLoader

from quoradataset import QuoraDataset
from quora_model import Quora_Sentence_BERT_Classifier

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

import torch
import torch.nn as nn
import os

def test(test_sentences_1,test_sentences_2,test_labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_encodings_1= tokenizer(test_sentences_2, truncation=True, padding=True,add_special_tokens=True)
    test_encodings_2= tokenizer(test_sentences_2, truncation=True, padding=True,add_special_tokens=True)
    test_labels=test_labels

    test_dataset = QuoraDataset(test_encodings_1,test_encodings_2, test_labels)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model=Quora_Sentence_BERT_Classifier()
    if os.path.exists('saved_model.pth'):
        model.load_state_dict(torch.load('saved_model.pth'))
        print("Saved Model Detected")
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    ground_truth = []
    y_predicted = []
    threshold=0.5
    for batch in test_loader:
        with torch.no_grad():
            input_ids_1= batch['input_ids'].to(device)
            attention_mask_1= batch['attention_mask'].to(device)
            input_ids_2= batch["input_ids_2"].to(device)
            attention_mask_2=batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2)
            predictions = outputs.squeeze()>threshold
            for i in labels.tolist():
                ground_truth.append(i)
            for i in predictions.tolist():
                y_predicted.append(i)

    accuracy = accuracy_score(ground_truth, y_predicted)

    print("F1 score is",f1_score(ground_truth, y_predicted))
    print("Precision  score is",precision_score(ground_truth, y_predicted))
    print("Recall score is",recall_score(ground_truth, y_predicted))
    print("The accuracy on Test Data is : ",accuracy)
    return accuracy,f1_score(ground_truth, y_predicted),precision_score(ground_truth, y_predicted),recall_score(ground_truth, y_predicted)

