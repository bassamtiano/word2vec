import sys
# Operasi matrix dengan cpu / gpu
import torch

# Operasi neural network
import torch.nn as nn

from utils.preprocessor import Preprocessor
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

class WordEmbedding(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size) -> None:
        super(WordEmbedding, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(1)
        
        # Matrix Transformation
        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.l2 = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def forward(self, x):
        out_bottleneck = self.l1(x)
        out = self.l2(out_bottleneck)

        out = self.softmax(out)
        return out, out_bottleneck

if __name__ == "__main__":

    max_length = 15
    batch_size = 100

    pre = Preprocessor(max_length = max_length, 
                       cased = False, 
                       wiki_dataset_dir = "datasets/ind_wikipedia_2021_1M-sentences.txt",
                       csv_dataset_dir = "datasets/Indonesian Sentiment Twitter Dataset Labeled.csv",
                       conllu_dataset_dir = "datasets/id_gsd-ud-train.conllu",
                       batch_size = int(batch_size))
    vocab_size, word2idx, idx2word, train_loader, valid_loader, test_loader, tokens = pre.preprocessor()

    training_data = np.empty((0, 2))
    # training_data = []

    window = 2

    for sentence in tqdm(tokens):
        # print(sentence)
        sent_len = len(sentence)
        # Potong kalimat jadi kata
        for i, word in enumerate(sentence):
            w_context = []
            # Check apakah text yang di proses adalah non padding
            if sentence[i] != 0:
                w_target = sentence[i]

                # i itu 0 - 14 karena i nya awal = 0 dan 14 itu maximum sentence length - 1
                for j in range(i - window, i + window + 1):
                    # Cek window
                    # j != i itu cek agar token target dan context tidak sama

                    if j != i and j <= sent_len - 1 and j >= 0 and sentence[j] != 0:
                        w_context = sentence[j]
                        # print(w_context)
                        training_data = np.append(training_data, [[w_target, w_context]], axis = 0)
                        # training_data.append([w_target, w_context])
                
    # print(len(training_data))
    # print(training_data.shape)

    # Window Pair untuk satu sentence
    # print(training_data.shape)
    # print(training_data[0:31])

    # print("ONE HOT")
    # Matrix kata untuk target di sentence
    enc = OneHotEncoder()
    enc.fit(np.array(range(vocab_size)).reshape(-1,1))
    onehot_label_x = enc.transform(training_data[:, 0].reshape(-1, 1).astype(int)).toarray()
    
    # print(training_data[:, 0].reshape(-1, 1))
    # categories = np.array(range(max_length)).reshape(-1,1)
    # print(categories)

    # print(type(training_data[0]))

    # Matrix kata untuk context di sentence
    enc_y = OneHotEncoder()
    enc_y.fit(np.array(range(vocab_size)).reshape(-1, 1))
    onehot_label_y = enc_y.transform(training_data[:, 1].reshape(-1, 1)).toarray()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WordEmbedding(input_size = vocab_size, hidden_size = window).to(device)
    model.train(True)

    onehot_label_x = torch.from_numpy(onehot_label_x).float().to(device)
    onehot_label_y = torch.from_numpy(onehot_label_y).float().to(device)

    # Loss and Optimizer function
    # BCE Loss untuk multi label dataset
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0, weight_decay = 0, nesterov = False)

    loss_val = []
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        for i in range(onehot_label_y.shape[0]):
            inputs = onehot_label_x[i]
            labels = onehot_label_y[i]

            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)

            output, wemb = model(inputs)

            loss = criterion(output, labels)
            print(loss.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        



    # Create tuple for training

