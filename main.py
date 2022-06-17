import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
import torchvision.datasets
import torchvision.transforms.functional as F

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


def cnn_model( pretrained=True, num_output=100):
    net = torchvision.models.resnet18(pretrained).to('cuda')
    net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_output))
    return net


model = cnn_model(True).to('cuda')
criterion = nn.CrossEntropyLoss().to('cuda')
opti = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-5)
scheduler = ReduceLROnPlateau(opti, mode='min', factor=0.1, 
                            patience=2, threshold=0.0001, threshold_mode='rel', 
                            cooldown=0, min_lr=0, eps=1e-08, verbose=False)
epoch = 20

for i in range(epoch):
    # Train
    acc, loss = train(model, train_dataloader, criterion, opti)
    train_acc.append(acc)
    train_loss.append(loss)
    
    # Validation
    acc, loss = evaluate(model, val_dataloader, criterion)
    val_acc.append(acc)
    val_loss.append(loss)
    
    
    scheduler.step(val_loss[-1])
    # Best Model save
    if pre_val < val_acc[-1]:
        model_save()a
        pre_val = val_acc[-1]
    
    # Print train and Validation Accuracy
    print(f"train: {train_acc[-1]} val: {val_acc[-1]}")
    print(f"train: {train_loss[-1]} val: {val_loss[-1]}")
    
y_pred = test(model, test_dataloader)

id_num = np.linspace(0, len(y_pred)-1, len(y_pred))

submission = pd.DataFrame({
    "Id" : id_num.astype(int),
    "Category" : y_pred.reshape(-1, ).astype(int)
})

submission.to_csv("result1.csv", index=False)
