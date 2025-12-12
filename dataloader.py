import fiftyone as fo
import fiftyone.zoo as foz
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms

# --------------------------
# Config
# --------------------------
# sampling only a subset of classes for faster experimentation  
desired_classes = [
    "person", "car", "bicycle", "dog", "cat",
    "bird", "chair", "couch", "laptop", "cell phone"
]
max_train = 40000
max_val = 10000

# --------------------------
# Load COCO via FiftyOne
# --------------------------
print("✅Loading COCO dataset from FiftyOne online ")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    classes=desired_classes,
   max_samples=max_train + max_val,
    download_if_necessary=True
)

print(f"Loaded dataset with {len(dataset)} images")

# --------------------------
# Split into train/test
# --------------------------
# FiftyOne provides `.shuffle()` and `.take()` methods
dataset.shuffle()  # shuffle images randomly
train_view = dataset.take(max_train)
test_view = dataset.skip(max_train).take(max_val)

print("✅Loading COCO dataset from FiftyOne done ! ")
# Label conversion helper
# --------------------------
class2idx = {cls: i for i, cls in enumerate(desired_classes)}

def labels_to_index(labels_list):
    """Return the integer label of the first desired class in labels_list"""
    for lbl in labels_list:
        if lbl in class2idx:
            return class2idx[lbl]
    return -1  # fallback if no label matches

# --------------------------
# PyTorch Dataset
# --------------------------
class FiftyOneCocoDataset(Dataset):
    def __init__(self, fo_view, transform=None):
        self.fo_view = fo_view
        self.transform = transform
        self.samples = list(fo_view)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # get labels
        labels = [obj.label for obj in sample.ground_truth.detections]
        label_idx = labels_to_index(labels)
        return img, label_idx

# --------------------------



batch_size = 128
transform = T.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    imgs, labels = zip(*batch)  # unzip images and labels
    imgs = default_collate(imgs)  # stack images into tensor
    labels = list(labels)          # keep labels as list of lists
    return imgs, labels

# --------------------------
# PyTorch DataLoaders
# --------------------------
print("✅Creating DataLoaders ")
train_dataset = FiftyOneCocoDataset(train_view, transform=transform)
val_dataset = FiftyOneCocoDataset(test_view, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True,persistent_workers = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True,persistent_workers = True)

print("✅Creating DataLoaders done ! ")



