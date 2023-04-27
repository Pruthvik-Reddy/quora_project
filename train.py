from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from quoradataset import QuoraDataset
from quora_model_base import Quora_Sentence_BERT_Classifier
import torch
import torch.nn as nn
import os

def train(train_sentences_1,train_sentences_2,train_labels,val_sentences_1,val_sentences_2,val_labels,
          batch_size,epochs,learning_rate):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings_1 = tokenizer(train_sentences_1, truncation=True, padding=True,add_special_tokens=True)
    val_encodings_1= tokenizer(val_sentences_1, truncation=True, padding=True,add_special_tokens=True)
    train_encodings_2= tokenizer(train_sentences_2, truncation=True, padding=True,add_special_tokens=True)
    val_encodings_2= tokenizer(val_sentences_2, truncation=True, padding=True,add_special_tokens=True)

    train_dataset = QuoraDataset(train_encodings_1,train_encodings_2, train_labels)
    val_dataset = QuoraDataset(val_encodings_1,val_encodings_2, val_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model=Quora_Sentence_BERT_Classifier()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    if os.path.exists('saved_model.pth'):
        model.load_state_dict(torch.load('saved_model.pth'))
        print("Saved Model Detected")
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    training_loss=0
    val_loss=0
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
            labels = batch['labels'].to(device)
            outputs = model(input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2)
            loss = loss_function(outputs.squeeze(),labels.to(torch.float))
            loss.backward()
            optim.step()
            print(cnt)
            training_loss+=loss.item()
            if cnt%100==0:
                torch.save(model.state_dict(), 'saved_model.pth')
            if cnt%1000==0:
                print("Epoch :",epoch+1)
                print(loss.item())
                print("Loss for batch : ",loss)
                #torch.save(model.state_dict(), 'model.pth')
        avg_training_loss=training_loss/len(train_loader)
        print("Training loss for epoch {} is {}".format(epoch+1,avg_training_loss))
        
        
        cnt2=0
        

        model.eval()
        dev_loss=0
        with torch.no_grad():
            
            for batch in val_loader:
                cnt2+=1
                input_ids_1= batch['input_ids'].to(device)
                attention_mask_1= batch['attention_mask'].to(device)
                input_ids_2= batch["input_ids_2"].to(device)
                attention_mask_2=batch['attention_mask_2'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2)
                loss = loss_function(outputs.squeeze(),labels.to(torch.float))
                dev_loss+=loss.item()
                if cnt2%1000==0:
                    print("Epoch : ",epoch)
                    print("Batch loss :",loss.item())
        torch.save(model.state_dict(), 'saved_model.pth')
        print("Dev loss for epoch {} is {}".format(epoch+1,dev_loss/len(val_loader)))
        
    model.eval()


