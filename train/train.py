import torch
from config.config import Config 
from torchvision import transforms
from torch.utils.data import DataLoader
from preprocessing.data_loader import FashionDataset
from models.resnet50 import get_model
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

def train_model(root_dir, num_classes, num_epochs=10, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    dataset = FashionDataset(root_dir=root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = get_model(num_classes)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # use cross entropy for now
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(data_loader)}')

    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
