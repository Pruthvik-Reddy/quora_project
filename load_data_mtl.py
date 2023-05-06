from sklearn.model_selection import train_test_split
import pandas as pd





def return_data():
    train_data=dict()
    test_data=dict()
    val_data=dict()

    train_df=pd.read_csv("quora_dataset/quora_duplicate_questions_train.tsv", sep='\t')
    test_df=pd.read_csv("quora_dataset/quora_duplicate_questions_test.tsv", sep='\t')
    
    train_sentences_1=[]
    train_sentences_2=[]
    test_sentences_1=[]
    test_sentences_2=[]
    
    train_labels_1=[]
    train_labels_2=[]
    test_labels=[]
    
    for index,row in train_df.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            label1=row["is_duplicate"]
            label2=row["is_same_category"]
            if sent1=="" and sent2=="":
                continue
            train_sentences_1.append(sent1)
            train_sentences_2.append(sent2)
            train_labels_1.append(label1)
            train_labels_2.append(label2)


    for index,row in test_df.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            label=row["is_duplicate"]
            if sent1=="" and sent2=="":
                continue
        
            test_sentences_1.append(sent1)
            test_sentences_2.append(sent2)
            test_labels.append(label)

    train_data["sentences_1"]=train_sentences_1
    train_data["sentences_2"]=train_sentences_2
    train_data["labels1"]=train_labels_1
    train_data["labels2"]=train_labels_2

    test_data["sentences_1"]=test_sentences_1
    test_data["sentences_2"]=test_sentences_2
    test_data["labels"]=test_labels

    

    return train_data,test_data
