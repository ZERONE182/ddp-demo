
import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments



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

def collate_fn(examples):
    values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x":values, "labels":labels}

# 定义模型
model = DemoModel()

# 定义数据集
# train_file_lines = sum(1 for line in open(args.train_data_file)
# train_dataset = CustomIterableDataset(args.train_data_file, train_file_lines)
# dev_dataset = CustomDataset(args.eval_data_file)
train_dataset = DemoDataset()

train_dataset_size = len(train_dataset)
num_train_epochs = 10
train_batch_size = 8
eval_batch_size = 8
# 计算整体迭代步数max_steps
total_training_samples = train_dataset_size * num_train_epochs
total_batch_size = train_batch_size * int(os.environ.get('WORLD_SIZE'))
max_steps = math.ceil(total_training_samples / total_batch_size)
# 构造TrainingArguments
train_args = TrainingArguments(
            output_dir='./test',
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            overwrite_output_dir=True,
            save_total_limit=1,
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
            learning_rate=1e-4,
            metric_for_best_model='eval_loss',
            fp16=True,
            max_steps=max_steps,
            dataloader_num_workers=10,
        )
# 构造Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=collate_fn,
)
# 开始炼丹!!
trainer.train()