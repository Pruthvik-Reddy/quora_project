import torch.nn as nn
from transformers import DistilBertModel
import torch

class Quora_Sentence_BERT_Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert=DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.dropout=nn.Dropout(0.1)
    self.task1_layer_1 = nn.Linear(768*3,1)
    self.task2_layer_2 = nn.Linear(768*3,1)
    #self.fully_connected_layer_2 = nn.Linear(256,1)
  
  def forward(self,input_ids_1,attention_mask_1,input_ids_2,attention_mask_2):
    u_embeddings=self.bert(input_ids_1,attention_mask=attention_mask_1)
    v_embeddings=self.bert(input_ids_2,attention_mask=attention_mask_2)
    u_pooled= u_embeddings[0].mean(dim=1)
    v_pooled = v_embeddings[0].mean(dim=1)
    u_v_subtracted = u_pooled - v_pooled
    concat_vector = torch.cat((u_pooled,v_pooled,u_v_subtracted),axis=1)
    probability_task_1 = self.task1_layer_1(concat_vector)
    probability_task_2 = self.task2_layer_2(concat_vector)

    #probability= self.fully_connected_layer_2(common_layer)
    return probability_task_1, probability_task_2 
