import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets
import torchvision.transforms.functional as F

from sklearn.model_selection import train_test_split



# GPU 캐시 비우기
gc.collect()
torch.cuda.empty_cache()

# 웹페이지에서 사진 그리기
plt.style.use('seaborn-white')
%matplotlib inline


data_load = np.load('cifar100_noising_train.npz')
cifar100_noisy_train_data = data_load['data']
cifar100_noisy_train_target = data_load['target']
print(cifar100_noisy_train_data.shape, cifar100_noisy_train_target.shape)

data_load = np.load('cifar100_noising_test.npz')
cifar100_noisy_test_public_data = data_load['data']
print(cifar100_noisy_test_public_data.shape)

cifar100_noisy_train_data, cifar100_noisy_val_data, cifar100_noisy_train_target, cifar100_noisy_val_target = \
train_test_split(cifar100_noisy_train_data, cifar100_noisy_train_target, test_size=0.1, shuffle=True, 
                 stratify=cifar100_noisy_train_target, random_state=34)

print(emnist_noisy_train_data.shape, emnist_noisy_train_target.shape)
print(emnist_noisy_val_data.shape, emnist_noisy_val_target.shape)

cifar100_noisy_train_data = torch.FloatTensor(cifar100_noisy_train_data)
cifar100_noisy_train_target = torch.LongTensor(cifar100_noisy_train_target)

emnist_noisy_val_data = torch.FloatTensor(emnist_noisy_val_data)
emnist_noisy_val_target = torch.LongTensor(emnist_noisy_val_target)

emnist_noisy_test_public_data = torch.FloatTensor(emnist_noisy_test_public_data)

emnist_noisy_train_data = F.resize(emnist_noisy_train_data, 72)
emnist_noisy_val_data = F.resize(emnist_noisy_val_data, 72)
emnist_noisy_test_public_data = F.resize(emnist_noisy_test_public_data, 72)

train_data = F.resize(emnist_noisy_train_data, 72)
val_data = F.resize(emnist_noisy_val_data, 72)
test_public_data = F.resize(emnist_noisy_test_public_data, 72)

train_dataset = TensorDataset(torch.FloatTensor(emnist_noisy_train_data),
                              torch.LongTensor(emnist_noisy_train_target))
val_dataset = TensorDataset(torch.FloatTensor(emnist_noisy_val_data),
                              torch.LongTensor(emnist_noisy_val_target))
test_dataset = TensorDataset(torch.FloatTensor(emnist_noisy_test_public_data),
                            torch.randint(0, 100, size=(emnist_noisy_test_public_data.shape[0],)))


batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
