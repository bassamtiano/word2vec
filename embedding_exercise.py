import sys

from utils.preprocessor import Preprocessor
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm


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
                
    print(len(training_data))
    print(training_data.shape)

    # Window Pair untuk satu sentence
    print(training_data.shape)
    print(training_data[0:31])

    # Matrix kata untuk target di sentence
    enc_x = OneHotEncoder()
    enc_x.fit(np.array(range(max_length)).reshape(-1,1))
    onehot_label_x = enc_x.transform(training_data[:, 0].reshape(-1, 1)).toarray()

    # print(onehot_label_x)

    # Matrix kata untuk context di sentence
    enc_y = OneHotEncoder()
    enc_y.fit(np.array(range(max_length)).reshape(-1, 1))
    onehot_label_y = enc_y.transform(training_data[:, 1].reshape(-1, 1)).toarray()

    


    # Create tuple for training

