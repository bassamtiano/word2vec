import sys
import torch.nn as nn

# Struktur class model pytorch

# Menprediksi berdasarkan konteks
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
            # 300
            max_norm = True
        )

        self.linear = nn.Linear(
            in_features = embed_dim,
            out_features = vocab_size
        )

    # Di eksekusi saat training
    def forward(self, inputs_):
        # developed embedding system dari kata -> embedding    

        # 1. Embedding cara kerja
        # 2. Matrix Projection Gimana ? 
        x = self.embeddings(inputs_)

        # Axis 1 = Horizontal / row-wise
        # Axis 0 = Vertical / column-wise
        x = x.mean(axis = 1)
        
        # Merubah bentuk dimensi x menjadi embedding dim (jumlah baris) x vocab size 
        x = self.linear(x)
        return x


# Menprediksi berdasarkan kata
class SkipGram(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_norm) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embedding = vocab_size,
            embedding_dim = embed_dim,
            max_norm = max_norm
        )

        self.linear = nn.Linear(
            in_features = embed_dim,
            out_features = vocab_size
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x