
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import CIFAR10CNN


# -----------------------------
# 1. Configuration
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./cifar_data"
MODEL_PATH = "./models/best_model.pth"
OUTPUT_DIR = "./outputs"

BATCH_SIZE = 64

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# 2. Data Preparation
# -----------------------------

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

# Test dataset
test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# -----------------------------
# 3. Load Trained Model
# -----------------------------

model = CIFAR10CNN().to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print(f"Device: {DEVICE}")
print("Version 1 model loaded successfully.")


# -----------------------------
# 4. Evaluate Test Dataset
# -----------------------------

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


test_loss = total_loss / total
test_accuracy = (correct / total) * 100

print("\n========== VERSION 1 TEST RESULTS ==========")
print(f"Test Images: {total}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {total - correct}")


# -----------------------------
# 5. Per-Class Performance
# -----------------------------

confusion_matrix = torch.zeros(
    10, 10, dtype=torch.int64
)

for actual, predicted in zip(all_labels, all_predictions):

    confusion_matrix[actual, predicted] += 1


print("\n========== CLASS-WISE RESULTS ==========")

for i, class_name in enumerate(CLASSES):

    true_positive = confusion_matrix[i, i].item()

    actual_count = confusion_matrix[i, :].sum().item()

    predicted_count = confusion_matrix[:, i].sum().item()

    class_accuracy = (
        true_positive / actual_count * 100
        if actual_count > 0 else 0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count > 0 else 0
    )

    recall = (
        true_positive / actual_count
        if actual_count > 0 else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    print(
        f"{class_name:12s} | "
        f"Accuracy: {class_accuracy:6.2f}% | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1_score:.3f}"
    )


# -----------------------------
# 6. Top 10 Classification Errors
# -----------------------------

error_matrix = confusion_matrix.clone()

# Remove correct predictions
error_matrix.fill_diagonal_(0)

top_values, top_indices = torch.topk(
    error_matrix.flatten(),
    k=10
)

print("\n========== TOP 10 CLASSIFICATION ERRORS ==========")

for count, index in zip(
    top_values.tolist(),
    top_indices.tolist()
):

    actual_class = index // 10
    predicted_class = index % 10

    print(
        f"Actual: {CLASSES[actual_class]:12s} -> "
        f"Predicted: {CLASSES[predicted_class]:12s} | "
        f"Count: {count}"
    )


# -----------------------------
# 7. Validation Data Preparation
# -----------------------------

print("\nGenerating validation logits...")

# Reproduce the original Phase 1 validation split
torch.manual_seed(42)

indices = torch.randperm(50000).tolist()

val_indices = indices[45000:]

val_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=test_transform
)

val_subset = Subset(
    val_dataset,
    val_indices
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# -----------------------------
# 8. Generate Validation Logits
# -----------------------------

model.eval()

all_val_logits = []
all_val_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        all_val_logits.append(outputs.cpu())

        all_val_labels.append(labels.cpu())


val_logits = torch.cat(all_val_logits)

val_labels = torch.cat(all_val_labels)


# Validation accuracy
val_predictions = torch.argmax(val_logits, dim=1)

val_accuracy = (
    val_predictions == val_labels
).float().mean().item() * 100

print("\n========== VALIDATION RESULTS ==========")
print(f"Validation Images: {len(val_labels)}")
print(f"Validation Accuracy: {val_accuracy:.2f}%")


# -----------------------------
# 9. Validation Confidence
# -----------------------------

val_probabilities = torch.softmax(
    val_logits,
    dim=1
)

val_confidences, val_predictions = torch.max(
    val_probabilities,
    dim=1
)

val_correct = val_predictions.eq(val_labels)

print("\n========== VALIDATION CONFIDENCE ==========")

print(
    f"Average Confidence: "
    f"{val_confidences.mean().item() * 100:.2f}%"
)

for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

    selected = val_confidences >= threshold

    count = selected.sum().item()

    if count > 0:

        accuracy = (
            val_correct[selected].float().mean().item() * 100
        )

        print(
            f"Confidence >= {threshold * 100:.0f}% | "
            f"Images: {count} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    else:

        print(
            f"Confidence >= {threshold * 100:.0f}% | "
            "No predictions"
        )


# -----------------------------
# 10. Save Validation Logits
# -----------------------------

VALIDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "v1_validation_logits.pth"
)

torch.save(
    {
        "logits": val_logits,
        "labels": val_labels
    },
    VALIDATION_PATH
)

print("\nValidation logits saved successfully.")
print("Logits shape:", val_logits.shape)
print("Labels shape:", val_labels.shape)


# -----------------------------
# 11. Plot Confusion Matrix
# -----------------------------

plt.figure(figsize=(10, 8))

plt.imshow(
    confusion_matrix.numpy(),
    cmap="Blues"
)

plt.title("CIFAR-10 Version 1 - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(10),
    CLASSES,
    rotation=45
)

plt.yticks(
    range(10),
    CLASSES
)

plt.colorbar()
plt.tight_layout()

matrix_path = os.path.join(
    OUTPUT_DIR,
    "v1_confusion_matrix.png"
)

plt.savefig(
    matrix_path,
    dpi=300
)

plt.show()

print(f"\nConfusion matrix saved: {matrix_path}")

print("\nVersion 1 evaluation completed.")
