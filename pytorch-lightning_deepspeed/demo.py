import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


ds = [torch.rand(1024*252*16) for i in range(128)]

class DemoModel(pl.LightningModule):
    def __init__(self,db):
        super().__init__()
        self.net = nn.Linear(1024*252*16, 1)
        self.db = db

    def forward(self, x):
        y = self.net(x)
        return y

    def train_dataloader(self):
        return DataLoader(self.db,batch_size=1)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = -(self(x)**2+2*x)[0]
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        return optimizer

model = DemoModel(ds)

# trainer = pl.Trainer(accelerator='ddp', gpus=8, num_nodes=2, max_epochs=100000)
trainer = pl.Trainer(gpus=2, num_nodes=2, strategy="deepspeed_stage_2", precision=16, max_epochs=100)
trainer.fit(model)
