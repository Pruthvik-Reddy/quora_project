import numpy as np
import umap
import hdbscan
import pandas as pd

embeddings=np.load("all_sentence_embeddings.npy")
print("The shape of embeddings before dmensionality reduction is : ",embeddings.shape )

reduced_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=10, 
                            metric='cosine').fit_transform(embeddings)

print("The shape of embeddings after dmensionality reduction is : ",reduced_embeddings.shape )


cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(reduced_embeddings)

labels=cluster.labels_
df = pd.DataFrame(labels, columns=["labels"])

df.to_csv("cluster_labels.csv", index=False)
df.to_csv("quora_dataset/cluster_labels.csv", index=False)

print("The dimensions of cluster labels is : ",df.shape)


