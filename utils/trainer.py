import sys

import torch
import torch.nn as nn

import pytorch_lightning as pl

from models.w2v import CBOW

class W2VTrainer(pl.LightningModule):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 embed_max_norm,
                 lr
                 ) -> None:

        super().__init__()

        self.cbow_model = CBOW(vocab_size,
                               embed_dim,
                               embed_max_norm)

        self.lr = lr
        self.criterion = nn.BCELoss()


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        # Standard Pytorch lightning itu train batch nya terdiri dari 2 item di yaitu x, y
        x = train_batch[0]
        
        output = self.cbow_model(x)
        print(output)


    # def validation_step(self, valid_batch, batch_idx)