import os   # files for file operations
from PIL import Image   # for image processing
from torchvision import transforms  # for transformations
from torch.utils.data import Dataset    # for data handling

# defining custom dataset class
class ImageDataset(Dataset):    # inherits from torch.utils.data.Dataset
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir    # for scanning root directory for subdirectories
        self.transform = transform
        self.image_paths = []   # for storing file path for each image
        self.labels = []    # for storing numeric labels (mapping folders to integers)
        self.label_map = {} # maps each label to its unique integer

        # load image paths and their labels (folder names)
        for label_idx, label in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, label)
            if os.path.isdir(folder_path):
                self.label_map[label_idx] = label  # label-to-index mapping
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)  # store the label as the folder index

    def __len__(self):  # returns number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx): # for loading and processing individual images when accessed
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# image transformations (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resizing all images to 128x128 pixels
    transforms.ToTensor(),  # convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize (ImageNet values)
])

# # create an instance of the dataset
# dataset = ImageDataset(root_dir='data/', transform=transform)

# # DataLoader for batch processing
# from torch.utils.data import DataLoader
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # print the size of the dataset
# print(f"Dataset size: {len(dataset)}")
