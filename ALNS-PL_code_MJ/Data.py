from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, x_0, x_1, y):
        self.x_0_data = x_0
        self.x_1_data = x_1
        self.y_data = y
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_0_data[index, :], self.x_1_data[index, :], self.y_data[index]

    def __len__(self):
        return self.length