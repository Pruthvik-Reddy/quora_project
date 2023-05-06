from sklearn.model_selection import train_test_split
import pandas as pd


data=pd.read_csv("quora_dataset/quora_duplicate_questions_with_categories.tsv", sep='\t')

train_df, test_df = train_test_split(data, test_size=0.4, random_state=42)

train_df.to_csv("quora_dataset/quora_duplicate_questions_train.tsv",sep="\t")

test_df.to_csv("quora_dataset/quora_duplicate_questions_test.tsv",sep="\t")
