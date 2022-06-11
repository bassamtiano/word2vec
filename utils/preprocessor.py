import pickle
import sys
import re
from tqdm import tqdm
import csv

from conllu import parse_incr

class Preprocessor():
    def __init__(self,
                 max_length,
                 wiki_dataset_dir,
                 csv_dataset_dir,
                 conllu_dataset_dir,
                 cased) -> None:

        # Object Class untuk vocab
        self.vocab = ['[PAD]']

        # Menentukan maksimal size token
        # Misal kurang tambahkan Padding
        # Misal lebih potong kata
        self.max_length = max_length
        self.cased = cased

        self.corpus = []
        # Corpus = Saya, Telah, Makan

        self.tokens = []

        self.wiki_dataset_dir = wiki_dataset_dir
        self.csv_dataset_dir = csv_dataset_dir
        self.conllu_dataset_dir = conllu_dataset_dir

    def clean_sentence(self, sentence):
        # Membersihkan dari karakter tidak standard
        sentence = re.sub(r"[^A-Za-z(),!?\'\`]", " ", sentence)


        sentence = re.sub(r"\'s", " \'s", sentence)
        sentence = re.sub(r"\'ve", " \'ve", sentence)
        sentence = re.sub(r"n\'t", " n\'t", sentence)
        sentence = re.sub(r"\n", "", sentence)
        sentence = re.sub(r"\'re", " \'re", sentence)
        sentence = re.sub(r"\'d", " \'d", sentence)
        sentence = re.sub(r"\'ll", " \'ll", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"'", "", sentence)
        sentence = re.sub(r'""', "", sentence)
        sentence = re.sub(r"\(", "", sentence)
        sentence = re.sub(r"\)", "", sentence)
        sentence = re.sub(r"\?", " \? ", sentence)
        sentence = re.sub(r"\,", "", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)

        if not self.cased:
            sentence = sentence.lower()

        return sentence.strip()

    def load_wiki(self):
        with open(self.wiki_dataset_dir, "r", encoding="utf-8") as rd:
            for i, line in enumerate(tqdm(rd)):
                item = line.split("\t")
                sentence = self.clean_sentence(item[1])
                self.prepare_corpus(sentence)

                if i > 10 :
                    break

    def load_twitter(self):
        with open(self.csv_dataset_dir, "r", encoding="utf-8") as f:
            next(f)
            rd = csv.reader(f, delimiter='t')
            for i, line in enumerate(tqdm(rd)):
                item = ''.join(line).split("\t")
                sentence = self.clean_sentence(item[1])
                self.prepare_corpus(sentence)

                if i > 10 :
                    break
    
    def load_conllu(self):
        with open(self.conllu_dataset_dir, "r", encoding="utf-8") as f:
            rd = parse_incr(f)
            for i, line in enumerate(tqdm(rd)):
                item = line.metadata['text']
                sentence = self.clean_sentence(item)
                self.prepare_corpus(sentence)

                if i > 10 :
                    break

    def preprocessor(self):
        self.load_wiki()
        self.load_twitter()
        self.load_conllu()
        
        self.word2idx = {w: idx for (idx, w) in enumerate(self.vocab)}
        self.idx2word = {idx: w for (idx, w) in enumerate(self.vocab)}

        vocabs = self.word2idx, self.idx2word

        print("Size Vocab ", len(self.vocab))
        with open("preprocessed/vocabs.pk", "wb") as handler:
            pickle.dump(vocabs, handler, protocol=pickle.HIGHEST_PROTOCOL)

        with open("preprocessed/corpus.pk", "wb") as handler:
            pickle.dump(self.corpus, handler, protocol=pickle.HIGHEST_PROTOCOL)

        self.tokenizer()


    def prepare_corpus(self, sentence):
        token = sentence.split(" ")
        # Create Vocab
        for t in token:
            if t not in self.vocab:
                self.vocab.append(t)

            
        # Create clean corpus
        self.corpus.append(token)

    def tokenizer(self):
        for cp in self.corpus:
            tkn = []
            for tk in cp:
                tkn.append(self.word2idx[tk])

            # Menyamakan panjang sentence
            if len(tkn) < self.max_length:
                tkn += [0] * (self.max_length - len(tkn))
            elif len(tkn) > self.max_length:
                tkn = tkn[:self.max_length]
            
            print("Size : ", len(tkn))
            
            self.tokens.append(tkn)
            

        print(len(self.tokens))
