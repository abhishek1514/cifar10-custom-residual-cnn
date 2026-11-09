
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os

# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. CIFAR-10 CLASSES
# ==========================================

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ==========================================
# 3. NORMALIZATION VALUES
# ==========================================

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)


# ==========================================
# 4. TRAINING TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(cifar_mean, cifar_std)
])


# ==========================================
# 5. VALIDATION AND TEST TRANSFORMATIONS
# ==========================================

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(cifar_mean, cifar_std)
])


# ==========================================
# 6. LOAD DATASETS
# ==========================================

train_dataset_aug = datasets.CIFAR10(
    root="./cifar_data",
    train=True,
    download=False,
    transform=train_transform
)

train_dataset_eval = datasets.CIFAR10(
    root="./cifar_data",
    train=True,
    download=False,
    transform=test_transform
)

test_dataset = datasets.CIFAR10(
    root="./cifar_data",
    train=False,
    download=False,
    transform=test_transform
)


# ==========================================
# 7. TRAIN-VALIDATION SPLIT
# ==========================================

torch.manual_seed(42)

indices = torch.randperm(len(train_dataset_aug)).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_dataset = Subset(
    train_dataset_aug,
    train_indices
)

val_dataset = Subset(
    train_dataset_eval,
    val_indices
)


# ==========================================
# 8. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ==========================================
# 9. DATASET INFORMATION
# ==========================================

print("\nDataset Information")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))


# ==========================================
# 10. VERIFY IMAGE TENSOR
# ==========================================

images, labels = next(iter(train_loader))

print("\nTensor Information")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())


# ==========================================
# 11. SAVE SAMPLE IMAGES
# ==========================================

os.makedirs("outputs", exist_ok=True)

# Load raw dataset for visualization
raw_dataset = datasets.CIFAR10(
    root="./cifar_data",
    train=True,
    download=False,
    transform=None
)

random_indices = random.sample(
    range(len(raw_dataset)), 10
)

plt.figure(figsize=(12, 6))

for i, idx in enumerate(random_indices):

    image, label = raw_dataset[idx]

    plt.subplot(2, 5, i + 1)

    plt.imshow(image)

    plt.title(classes[label])

    plt.axis("off")

plt.tight_layout()

plt.savefig("outputs/sample_images.png")

plt.show()

print("\nPhase 1 data preparation completed.")
