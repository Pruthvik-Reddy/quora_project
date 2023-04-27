from sklearn.model_selection import train_test_split
import pandas as pd





def return_data():
    train_data=dict()
    test_data=dict()
    val_data=dict()

    data=pd.read_csv('quora_dataset/quora_duplicate_questions.tsv', sep='\t',nrows=50000)
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    
    train_sentences_1=[]
    train_sentences_2=[]
    test_sentences_1=[]
    test_sentences_2=[]
    val_sentences_1=[]
    val_sentences_2=[]

    train_labels=[]
    test_labels=[]
    val_labels=[]

    for index,row in train_df.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            label=row["is_duplicate"]
            if sent1=="" and sent2=="":
                continue
            train_sentences_1.append(sent1)
            train_sentences_2.append(sent2)
            train_labels.append(label)

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

    for index,row in val_df.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
        
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            label=row["is_duplicate"]
            if sent1=="" and sent2=="":
                continue
        
            val_sentences_1.append(sent1)
            val_sentences_2.append(sent2)
            val_labels.append(label)

    train_data["sentences_1"]=train_sentences_1
    train_data["sentences_2"]=train_sentences_2
    train_data["labels"]=train_labels

    test_data["sentences_1"]=test_sentences_1
    test_data["sentences_2"]=test_sentences_2
    test_data["labels"]=test_labels

    val_data["sentences_1"]=val_sentences_1
    val_data["sentences_2"]=val_sentences_2
    val_data["labels"]=val_labels


    return train_data,test_data,val_data
