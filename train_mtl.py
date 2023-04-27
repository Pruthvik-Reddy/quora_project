from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from quora_dataset_mtl import QuoraDataset
from quora_model_mtl import Quora_Sentence_BERT_Classifier
import torch
import torch.nn as nn
import os

def train(train_sentences_1,train_sentences_2,train_labels_1, train_labels_2,
          batch_size, epochs, learning_rate):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings_1 = tokenizer(train_sentences_1, truncation=True, padding=True,add_special_tokens=True)
    train_encodings_2= tokenizer(train_sentences_2, truncation=True, padding=True,add_special_tokens=True)
    train_dataset = QuoraDataset(train_encodings_1,train_encodings_2, train_labels_1,train_labels_2)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model=Quora_Sentence_BERT_Classifier()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    if os.path.exists('saved_model_mtl.pth'):
        model.load_state_dict(torch.load('saved_model_mtl.pth'))
        print("Saved Model for MTL Detected")
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    
    training_loss=0
    cnt=0
    for epoch in range(epochs):
        model.train()
        training_loss=0
        for batch in train_loader:
            cnt+=1
            optim.zero_grad()
            input_ids_1= batch['input_ids'].to(device)
            attention_mask_1= batch['attention_mask'].to(device)
            input_ids_2= batch["input_ids_2"].to(device)
            attention_mask_2=batch['attention_mask_2'].to(device)
            labels_1 = batch['labels1'].to(device)
            labels_2 = batch['labels2'].to(device)
            outputs_1,outputs_2 = model(input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2)
            loss_1= loss_function(outputs_1.squeeze(),labels_1.to(torch.float))
            loss_2= loss_function(outputs_2.squeeze(),labels_2.to(torch.float))
            loss=(0.8*loss_1)+(0.2*loss_2)
            loss.backward()
            optim.step()
            print(cnt)
            training_loss+=loss.item()
            if cnt%100==0:
                torch.save(model.state_dict(), 'saved_model_mtl.pth')
            if cnt%1000==0:
                print("Epoch :",epoch+1)
                print(loss.item())
                print("Loss for batch : ",loss)
                #torch.save(model.state_dict(), 'model.pth')
        avg_training_loss=training_loss/len(train_loader)
        print("Training loss for epoch {} is {}".format(epoch+1,avg_training_loss))
        
        
        

        
    model.eval()
    

