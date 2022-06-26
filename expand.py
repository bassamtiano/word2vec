from conllu import parse_incr

def expand_dataset():
    with open("datasets/id_gsd-ud-train.conllu", "r", encoding="utf-8") as f:
        rd = parse_incr(f)    
        for line in rd:
            print(line.metadata["text"])
            break

if __name__ == "__main__":
    expand_dataset()