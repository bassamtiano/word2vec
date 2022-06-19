import sys
import torch.nn as nn

# Struktur class model pytorch
class CBOW(nn.Module):
    # Dieksekusi saat di deklarasikan
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 embed_max_norm) -> None:
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim,
            max_norm = True
        )

        self.linear = nn.Linear(
            in_features = embed_dim,
            out_features = vocab_size
        )

    # Di eksekusi saat training
    def forward(self, inputs_):
        # developed embedding system dari kata -> embedding        
        x = self.embeddings(inputs_)

        x = x.mean(axis = 1)
        x = self.linear(x)
        return x


# class SkipGram(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, inputs_):
#         # Disini