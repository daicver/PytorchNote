import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        # 可以在这里对图像进行必要的预处理，例如转换为Tensor
        image = torch.tensor(image, dtype=torch.float32)
        return image, label

data, labels = list, list
custom_dataset = CustomDataset(data, labels)

# 创建数据加载器，指定批量大小和其他参数
batch_size = 64
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    images, labels = batch