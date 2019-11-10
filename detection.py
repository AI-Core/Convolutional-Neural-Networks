import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import S40dataset
import matplotlib.pyplot as plt
import time

# HYPERPARAMETERS
batch_size = 4
epochs = 10
lr = 0.0001

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# MAKE DATASET
dataset = S40dataset.S40dataset(transform=transform)

# SPLIT DATASET
train_len = int( 0.8 * len(dataset))
val_len = int( 0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

# MAKE DATALOADERS
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_data,
                         batch_size=batch_size)
for i in test_loader:
    b = i
    break
import numpy as np
idx = np.random.randint(len(test_loader))
S40dataset.show(b)
k

print('GPU available:', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


channels1 = 128
channels2 = 64
channels3 = 64
channels4 = 64

class DetectCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, channels1, 7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels1),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(channels1, channels2, 7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(channels2, channels3, 7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(channels3)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*46*46, 4),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)

        x = x.view(-1, 64*46*46)
        x = self.fc(x)
        return x

cnn = DetectCNN()
cnn.to(device)

criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(params=cnn.parameters(), lr=lr, weight_decay=1)

def train(model):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()
    #plt.ylim(0, 1)

    train_losses = []
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            img, bndbox = batch
            img, bndbox = img.to(device), bndbox.to(device)
            pred_bndbox = model(img)
            print('label:', bndbox, 'prediction:', pred_bndbox)
            loss = criterion(pred_bndbox, bndbox)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print('Epoch:', epoch, 'Batch:', batch_idx, 'Loss:', loss.item())
            plt.cla()
            train_losses.append(loss.item())
            ax.plot(train_losses)
            fig.canvas.draw()
            if batch_idx == 200:
                #break
                pass

train(cnn)
torch.save(cnn.state_dict(), str(time.time()))

def test(model):
    model.eval()
    for idx, batch in enumerate(test_loader):
        print(type(batch))
        x, y = batch
        pred_bndbox = model(x)
        S40dataset.show(batch, pred_bndbox=pred_bndbox)
        if idx == 10:
            break


test(cnn)