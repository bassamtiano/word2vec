from utils.preprocessor import Preprocessor

if __name__ == "__main__":
    pre = Preprocessor(max_length = 100, 
                       cased = False, 
                       wiki_dataset_dir = "datasets/ind_wikipedia_2021_1M-sentences.txt",
                       csv_dataset_dir = "datasets/Indonesian Sentiment Twitter Dataset Labeled.csv")
    pre.preprocessor()

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
