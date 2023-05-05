import pandas as pd

# assume df is the original dataframe with 400,000 samples
data=pd.read_csv('quora_dataset/quora_duplicate_questions_preprocessed.tsv', sep='\t')
labels_df=pd.read_csv("quora_dataset/cluster_labels.csv")
labels=labels_df["labels"].to_list()
row_num=0
def label_data_with_categories():
    id1_list=[]
    id2_list=[]
    questions_1=[]
    questions_2=[]
    categories1=[]
    categories2=[]
    is_same_category=[]
    is_duplicate=[]
    row_num=0

    for index,row in data.iterrows():
        if isinstance(row["question1"],str) and isinstance(row["question2"],str):
            id1=row["qid1"]
            id2=row["qid2"]
            sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
            if sent1=="" and sent2=="":
                continue
            
            id1_list.append(id1)
            id2_list.append(id2)
            questions_1.append(sent1)
            questions_2.append(sent2)
            categories1.append(labels[row_num])
            categories2.append(labels[row_num+1])
            if labels[row_num]==labels[row_num+1]:
                is_same_category.append(1)
            else:
                is_same_category.append(0)
            is_duplicate.append(row["is_duplicate"])
            row_num+=2
    print("Id 1 length : ",len(id1_list))
    print("Id 2 length : ",len(id2_list))
    print("Question 1 length : ",len(questions_1))
    print("Question 2 length : ",len(questions_2))
    print("categories 1 length : ",len(categories1))
    print("Categories 2 length : ",len(categories2))
    print("Is same category length : ",len(is_same_category))
    
    data_with_categories = pd.DataFrame({'qid1': id1_list, 'qid2': id2_list, 'question1': questions_1, 'question2': questions_2,
                                         "categories1":categories1,"categories2":categories2,"is_same_category":is_same_category,"is_duplicate":is_duplicate})
    
    return data_with_categories

data_categories=label_data_with_categories()
data_categories.to_csv("quora_dataset/quora_duplicate_questions_with_categories.tsv",sep="\t")
