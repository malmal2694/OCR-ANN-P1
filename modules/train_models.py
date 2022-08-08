from .dataset import OCRDataset
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize


class TrainModel:
    def __init__(self, model):
        self.model = model()
        dataset = OCRDataset("create-data/data/", transforms=Compose([Resize((200, 2000))]))
        self.dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    def fit(self, opt_lr, num_epoch=2):
        optimizer = optim.Adam(self.model.parameters(), lr=opt_lr)
        loss_fn = CTCLoss()
        
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                imgs = data["img"]
                gts = data["gt"]
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(imgs)
                loss = loss_fn(outputs, gts)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 5 == 0:
                    print(f"Iteration {i+1} of epoch {epoch}) loss: {running_loss / 5:.4f} ")
                    running_loss = 0