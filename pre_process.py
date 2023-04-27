
from sklearn.model_selection import train_test_split

# assume df is the original dataframe with 400,000 samples
data=pd.read_csv('quora_dataset/quora_duplicate_questions.tsv', sep='\t')
train_size = 20000
test_size = 10000
random_state = 42

# randomly split the data into training and test sets
train, test = train_test_split(data, train_size=train_size, test_size=test_size, random_state=random_state)

# print the number of samples in each set
print("Number of training samples:", len(train))
print("Number of test samples:", len(test))

def pre_process(data):
  id1_list=[]
  id2_list=[]
  questions_1=[]
  questions_2=[]

  for index,row in data.iterrows():
    if isinstance(row["question1"],str) and isinstance(row["question2"],str):
      id1=row["qid1"]
      id2=row["qid2"]
      sent1=row["question1"].replace("\r", "").replace("\n", " ").replace("\t", " ")
      sent2=row["question2"].replace("\r", "").replace("\n", " ").replace("\t", " ")
      if sent1=="" and sent2=="":
          continue
      
      questions_1.append(row["question1"])
      questions_2.append(row["question2"])
      id1_list.append(id1)
      id2_list.append(id2)

  sub_sampled_data = pd.DataFrame({'qid1': id1_list, 'qid2': id2_list, 'question1': questions_1, 'question2': questions_2})
  return sub_sampled_data

sub_sampled_train=pre_process(train)
sub_sampled_test=pre_process(test)
sub_sampled_train.to_csv("sub_sampled_train.tsv",sep="\t",index=False)
sub_sampled_test.to_csv("sub_sampled_test.tsv",sep="\t",index=False)
