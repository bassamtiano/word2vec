from utils.preprocessor import Preprocessor
import pytorch_lightning as pl
from utils.trainer import W2VTrainer

import argparse

def parse_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embed_dim", help="Dimensi dari Embedding", required = True)
    parser.add_argument("--embed_max_norm", help="Maksimal Normal Embedding", required = True)
    parser.add_argument("--batch_size", help="Jumlah Batch", required = True)
    parser.add_argument("--lr", help="Jumlah Batch", required = True)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_parameter()

    pre = Preprocessor(max_length = 50, 
                       cased = False, 
                       wiki_dataset_dir = "datasets/ind_wikipedia_2021_1M-sentences.txt",
                       csv_dataset_dir = "datasets/Indonesian Sentiment Twitter Dataset Labeled.csv",
                       conllu_dataset_dir = "datasets/id_gsd-ud-train.conllu",
                       batch_size = int(args.batch_size))
    vocab_size, word2idx, idx2word, train_loader, valid_loader, test_loader = pre.preprocessor()

    trainer_runner = W2VTrainer(
        vocab_size = vocab_size,
        embed_dim = int(args.embed_dim),
        embed_max_norm = 1,
        lr = float(args.lr)
    )

    trainer = pl.Trainer(max_epochs = 10, default_root_dir = "./checkpoints")
    trainer.fit(trainer_runner, train_loader, valid_loader)






    

    # cleaned = pre.clean_sentence("Apakah ini adalah kalimat?")
    # print(cleaned)

    # sent = ["this is shit", "i eat shit"]
    
    # vocab = []
    # for s in sent:
    #     token = s.split(" ")
    #     for t in token:
    #         if t not in vocab:
    #             vocab.append(t)

    # print(vocab)
