import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


pizzapath = "./pizza.npy"
notpizzapath = "./not_pizza.npy"

class Dataset(Dataset):
    def __init__(self, pizzapath, notpizzapath, train):
        self.pizzas = np.load(pizzapath)
        self.real_test = self.pizzas[-train:]
        self.pizzas = self.pizzas[:-train]
        self.notpizzas = np.load(notpizzapath)
        self.fake_test = self.notpizzas[-train:]
        self.notpizzas = self.notpizzas[:-train]

    def __len__(self):
        return len(self.pizzas)

    def __getitem__(self, idx):
        pizza = self.pizzas[idx].astype(np.float32) / 127.5 - 1.0
        notpizza = self.notpizzas[idx].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(pizza), torch.from_numpy(notpizza)

    def get_test(self):
        return torch.from_numpy(self.real_test.astype(np.float32) / 127.5 - 1.0), torch.from_numpy(self.fake_test.astype(np.float32) / 127.5 - 1.0)



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # start at (3, 384, 512)
        self.conv_layer1 = nn.Sequential(
            # (64, 96, 128)
            nn.Conv2d(3, 64, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_layer2 = nn.Sequential(
            # (128, 24, 32)
            nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout2d(0.3)
        )
        # self.conv_layer3 = nn.Sequential(
        #     # (256, 48, 64)
        #     nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(0.2, inplace=False),
        # )
        # self.conv_layer4 = nn.Sequential(
        #     # (512, 24, 32)
        #     nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(0.2, inplace=False),
        # )
        # self.conv_layer5 = nn.Sequential(
        #     # (1028, 12, 16)
        #     nn.Conv2d(512, 1028, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(0.2, inplace=False),
        #     nn.Dropout2d(0.3)
        # )


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*24*32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        return self.fc(x)


def show(img):
    # Display image
    img = (img + 1)*127.5
    img = img.clip(0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap='grey', vmin=0, vmax=255)
    plt.show()


def save_checkpoint(dis, dis_opt, epoch, path):
    torch.save({
        'epoch': epoch,
        'dis_state_dict': dis.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }, path)


def load_checkpoint(dis, dis_opt, path):
    if os.path.exists(path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        dis.load_state_dict(checkpoint['dis_state_dict'])
        dis_opt.load_state_dict(checkpoint['dis_optimizer'])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    return 0


def trainNN(epochs=0, batch_size=16, lr=0.0002, save_time=1, save_dir='', device='cuda' if torch.cuda.is_available() else 'cpu'):
    dis = Classifier().to(device)
    criterion = torch.nn.BCELoss()
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = load_checkpoint(dis, dis_opt, save_dir)


    dataset = Dataset(pizzapath, notpizzapath, 50)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        for pizzas, nonpizzas in loader:
            pizzas = pizzas.to(device, non_blocking=True)
            nonpizzas = nonpizzas.to(device, non_blocking=True)

            dis_opt.zero_grad()

            real_preds = dis(pizzas)
            fake_preds = dis(nonpizzas)

            real_labels = (torch.ones_like(real_preds)).to(device)
            fake_labels = (torch.zeros_like(fake_preds)).to(device)

            real_loss = criterion(real_preds, real_labels)
            fake_loss = criterion(fake_preds, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            dis_opt.step()


        if (epoch + 1) % save_time == 0:
            save_checkpoint(dis, dis_opt, epoch + 1, save_dir)
            print(f"Epoch {epoch + 1} - real Loss: {real_loss.item():.4f}, fake Loss: {fake_loss.item():.4f}")


    real_test, fake_test = dataset.get_test()
    real_test = real_test.to(device)
    fake_test = fake_test.to(device)
    real_preds = torch.round(dis(real_test))
    fake_preds = torch.round(dis(fake_test))
    real_labels = torch.ones_like(real_preds)
    fake_labels = torch.zeros_like(fake_preds)

    y_true = np.concatenate([real_labels.cpu().detach().numpy(), fake_labels.cpu().detach().numpy()])
    y_pred = np.concatenate([real_preds.cpu().detach().numpy(), fake_preds.cpu().detach().numpy()])

    cm = confusion_matrix(y_true, y_pred)
    disp_cm = ConfusionMatrixDisplay(cm)
    disp_cm.plot()
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Test Images and Predictions", fontsize=16)


    num_show = 5
    real_imgs = real_test[:num_show].cpu().detach().numpy()
    fake_imgs = fake_test[:num_show].cpu().detach().numpy()
    real_preds_np = real_preds[:num_show].cpu().detach().numpy()
    fake_preds_np = fake_preds[:num_show].cpu().detach().numpy()

    for i in range(num_show):
        img = real_imgs[i]
        pred = "Real" if real_preds_np[i] >= 0.5 else "Fake"
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Pred: {pred}")
        axes[0, i].axis("off")

    for i in range(num_show):
        img = fake_imgs[i]
        pred = "Real" if fake_preds_np[i] >= 0.5 else "Fake"
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Pred: {pred}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real Test", fontsize=12)
    axes[1, 0].set_ylabel("Fake Test", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    print("CUDA Available:", torch.cuda.is_available())
    trainNN(0, 16, save_time=1, save_dir='save2.pth')