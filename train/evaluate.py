import torch
from torch.utils.data import DataLoader
from preprocessing.data_loader import FashionDataset
from models.resnet50 import get_model
from torchvision import transforms

def evaluate_model(root_dir, num_classes, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FashionDataset(root_dir=root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(num_classes)
    model.load_state_dict(torch.load('fashion_model.pth'))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')
