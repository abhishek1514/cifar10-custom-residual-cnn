
import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import CIFAR10CNN
from data_preparation import train_loader, val_loader, device


# ==========================================

# ==========================================

EPOCHS = 25
LEARNING_RATE = 0.001

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================

# ==========================================

model = CIFAR10CNN().to(device)

print("Model loaded on:", device)


# ==========================================

# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 4. CHECKPOINT CONFIGURATION
# ==========================================

checkpoint_path = "models/latest_checkpoint.pth"
best_model_path = "models/best_model.pth"

best_accuracy = 0.0
start_epoch = 0

# Set True to resume interrupted training
RESUME = False

if RESUME and os.path.exists(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_accuracy = checkpoint["best_accuracy"]

    print("Resuming from epoch:", start_epoch)


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        
        optimizer.zero_grad()

        
        outputs = model(images)

       
        loss = criterion(outputs, labels)

        
        loss.backward()

        
        optimizer.step()

        running_loss += loss.item()

        
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy


# ==========================================
# 7. MAIN TRAINING LOOP
# ==========================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

for epoch in range(start_epoch, EPOCHS):

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    # Training
    train_loss, train_accuracy = train_one_epoch()

    # Validation
    val_loss, val_accuracy = validate()

    # Store metrics
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    # Save latest checkpoint after every epoch
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": max(best_accuracy, val_accuracy),
        "history": history
    }

    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model saved!")

    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


# ==========================================
# 8. TRAINING COMPLETED
# ==========================================

print("\nTraining Completed!")

print(
    f"Best Validation Accuracy: {best_accuracy:.2f}%"
)

print("Latest checkpoint:", checkpoint_path)

print("Best model:", best_model_path)

# Save training history
torch.save(
    history,
    "outputs/training_history.pth"
)

print("Training history saved!")
