import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import PartialState
from torch.utils.data import DataLoader, Dataset


class DemoDataset(Dataset):
    def __init__(self,):
        super().__init__()
        self.db_i = [torch.rand(1024*252*16) for i in range(128)]
        self.db_o = [torch.rand(1) for _ in range(128)]
    
    def __getitem__(self, idx):
        return self.db_i[idx], self.db_o[idx]
    
    def __len__(self):
        return 128

class DemoModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = nn.Linear(1024*252*16, 1)

    def forward(self, x):
        y = self.net(x)
        return y

accelerator = Accelerator()
device = accelerator.device

model = DemoModel()
dataset = DemoDataset()
optimizer = torch.optim.Adam(model.parameters())

data = torch.utils.data.DataLoader(dataset, batch_size=8)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
    loss_total = 0
    num_elems = 0
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.mse_loss(output, targets)

        accelerator.backward(loss)

        optimizer.step()

        # loss_value = loss.item()
        # loss_total += loss_value
        accelerator_loss = accelerator.gather(loss)
        accelerator_output = accelerator.gather(output)
        num_elems += accelerator_output.shape[0]

        # accelerator.print(accelerator_loss.shape)
        accelerator.print(f"num_elems: {num_elems} 4_device_avg_loss: {accelerator_loss.mean()}")


# 等待所有进程达到一定程度后再执行指令
accelerator.wait_for_everyone()
# 只在主进程中保存模型
state = PartialState()
if state.is_main_process:
    print("Save model in main process...")
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model,'./test.pth')