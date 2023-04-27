import torch



class QuoraDataset(torch.utils.data.Dataset):
    def __init__(self, encodings1,encodings2, labels):
        self.sent1_encodings = encodings1
        self.sent2_encodings = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.sent1_encodings.items()}
        item.update({key+'_2': torch.tensor(val[idx]) for key, val in self.sent2_encodings.items()})
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)