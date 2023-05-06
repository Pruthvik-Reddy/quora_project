# quora_project
The Goal of this project is to improve accuracy on detecting if two questions are similar. This helps in de-cluttering platforms like quora,stacoverflow,etc.  
The Baseline is Sentence BERT which is built from scratch. 
All the files main.py,train.py,test.py are related to Baseline. 
To train -> python3 main.py --mode train
To test -> python3 main.py --mode test

Muti-task learning model : 
1. The secondary task is to detect if the pair of questions belong to the same category. 
2. But it does not have training data, so used unsupervised learning to get labels. 
3. From pre-trained sentence bert model "distilbert-base-nli-stsb-quora-ranking", extracted sentence embeddings of all files. The embeddings are stored in.npy file
4. Once sentence embeddings are obtained, it was passed through dimensionality reduction. 
5. UMAP is used for dimensionality reduction over PCA. UMAP (Uniform Manifold Approximation and Projection) is a nonlinear technique that is particularly
good at preserving the local structure of the data, while still allowing for global structure to be retained. Both PCA and UMAP can be used for reducing the
dimensions of BERT embeddings. However, UMAP is often preferred because it can preserve the local structure of high-dimensional data better than PCA, making it 
more suitable for visualizing and clustering embeddings. 
6. Clustering is done using HDBSCAN. HDBSCAN is a density-based algorithm that works quite well with UMAP since UMAP maintains a lot of local structure even in 
lower-dimensional space
7. The cluster label for each question is stored in cluster_labels.csv
8. The weight factor for updating weights for two tasks is taken as 0.6 (primary weight factor) and 0.4 (secondary weight factor) while calculating total_loss. 

Multi Task Learning Model : 

All the files which end _mtl are related to multi-task learning. 
To train : python3 main_mtl.py --mode train
To test : python3 main_mtl.py --mode test

Note about the datasets : 
1. quora_duplicate_questions.tsv - Original file with columns ( qid1, qid2, question 1 ,question 2, is_duplicate)
2. cluster_labels.csv - Contains cluster label for each question in the dataset. 
3. quora_duplicate_questions_preprocessed.tsv - From the original dataset, removed all rows which are empty or one question is empty or is not string. 
4. quora_duplicate_questions_with_categories.tsv - For the original file, added three columns ( categories 1, categories 2, is_same_category) based on labels from cluster_labels.csv
5. quora_duplicate_questions_train and quora_duplicate_questions_test are the train_test_split of quora_duplicate_questions_with_categories ( 60:40 ratio)
6. Remaining files are sub-sampling files so that training is done on a small sample of dataset with time constraint in mind. 
