
import os
import torch
import torch.nn as nn
import torch.optim as optim

from model_v2 import CIFAR10CNN_V2
from data_preparation import train_loader, val_loader, device


# --------------------------------

# --------------------------------

EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
PATIENCE = 10

MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"

LATEST_CHECKPOINT = os.path.join(
    MODEL_DIR, "latest_checkpoint_v2.pth"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_model_v2.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR, "v2_training_history.pth"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", device)


# --------------------------------
# 2. Initialize Model
# --------------------------------

model = CIFAR10CNN_V2().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# --------------------------------
# 3. Training History
# --------------------------------

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "learning_rate": []
}

best_val_accuracy = 0.0
best_val_loss = float("inf")

start_epoch = 0
epochs_without_improvement = 0

# Change to True only when intentionally resuming
RESUME = False


# --------------------------------
# 4. Resume Training
# --------------------------------

if RESUME and os.path.exists(LATEST_CHECKPOINT):

    print("\nResuming Version 2 training...")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_val_accuracy = checkpoint["best_val_accuracy"]
    best_val_loss = checkpoint["best_val_loss"]

    epochs_without_improvement = checkpoint[
        "epochs_without_improvement"
    ]

    history = checkpoint["history"]

    print("Resuming from epoch:", start_epoch + 1)

elif RESUME:

    raise FileNotFoundError(
        "Latest Version 2 checkpoint not found."
    )


# --------------------------------
# 5. Train One Epoch
# --------------------------------

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = (correct / total) * 100

    return epoch_loss, epoch_accuracy


# --------------------------------
# 6. Validation
# --------------------------------

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    val_loss = running_loss / total

    val_accuracy = (correct / total) * 100

    return val_loss, val_accuracy


# --------------------------------
# 7. Main Training Loop
# --------------------------------

print("\n========== CIFAR-10 VERSION 2 TRAINING ==========")

for epoch in range(start_epoch, EPOCHS):

    current_lr = optimizer.param_groups[0]["lr"]

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    # Record history
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    history["learning_rate"].append(current_lr)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(f"Learning Rate: {current_lr:.7f}")

    # Save best model based on validation accuracy
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Best Version 2 model saved!")

    # Early stopping monitors validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # Update learning rate
    scheduler.step()

    # Save latest checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history
        },
        LATEST_CHECKPOINT
    )

    # Save training history
    torch.save(
        history,
        HISTORY_PATH
    )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# --------------------------------
# 8. Training Completed
# --------------------------------

print("\n========== VERSION 2 TRAINING COMPLETED ==========")

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

print(f"Best Validation Loss: {best_val_loss:.4f}")

print("Best Model:", BEST_MODEL_PATH)

print("Latest Checkpoint:", LATEST_CHECKPOINT)

print("Training History:", HISTORY_PATH)
