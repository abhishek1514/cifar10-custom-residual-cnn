
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# --------------------------------
# 1. Configuration
# --------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_PATH = "./outputs/v1_validation_logits.pth"
OUTPUT_DIR = "./outputs"
TEMPERATURE_PATH = "./models/v1_temperature.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------
# 2. Load Validation Logits
# --------------------------------

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        "Validation logits not found. Run evaluate.py first."
    )

data = torch.load(
    INPUT_PATH,
    map_location="cpu",
    weights_only=True
)

logits = data["logits"].float()
labels = data["labels"].long()

print("Validation logits loaded.")
print("Logits shape:", logits.shape)
print("Labels shape:", labels.shape)


# --------------------------------
# 3. Split Calibration and Evaluation Data
# --------------------------------

# Keep half for fitting temperature
# Use the other half to evaluate calibration

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(labels),
    generator=generator
)

split = len(labels) // 2

fit_indices = indices[:split]
eval_indices = indices[split:]

fit_logits = logits[fit_indices].to(DEVICE)
fit_labels = labels[fit_indices].to(DEVICE)

eval_logits = logits[eval_indices].to(DEVICE)
eval_labels = labels[eval_indices].to(DEVICE)

print("\nCalibration samples:", len(fit_labels))
print("Calibration evaluation samples:", len(eval_labels))


# --------------------------------
# 4. Define Temperature Scaling
# --------------------------------

class TemperatureScaling(nn.Module):

    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(
            torch.ones(1, device=DEVICE)
        )

    def forward(self, logits):

        temperature = self.temperature.clamp(min=0.05)

        return logits / temperature


calibrator = TemperatureScaling().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.LBFGS(
    calibrator.parameters(),
    lr=0.01,
    max_iter=100
)


# --------------------------------
# 5. Optimize Temperature
# --------------------------------

def closure():

    optimizer.zero_grad()

    scaled_logits = calibrator(fit_logits)

    loss = criterion(scaled_logits, fit_labels)

    loss.backward()

    return loss


optimizer.step(closure)

temperature = calibrator.temperature.item()

print("\n========== TEMPERATURE SCALING ==========")
print(f"Learned Temperature: {temperature:.4f}")


# --------------------------------
# 6. Calculate Evaluation Metrics
# --------------------------------

def calculate_ece(probabilities, labels, bins=15):

    confidences, predictions = probabilities.max(dim=1)

    accuracies = predictions.eq(labels).float()

    ece = torch.tensor(0.0, device=DEVICE)

    bin_edges = torch.linspace(
        0, 1, bins + 1, device=DEVICE
    )

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        if mask.any():

            avg_confidence = confidences[mask].mean()
            avg_accuracy = accuracies[mask].mean()

            ece += mask.float().mean() * torch.abs(
                avg_accuracy - avg_confidence
            )

    return ece.item()


def calculate_metrics(logits, labels):

    probabilities = torch.softmax(logits, dim=1)

    predictions = probabilities.argmax(dim=1)

    accuracy = predictions.eq(labels).float().mean().item()

    nll = nn.functional.cross_entropy(
        logits, labels
    ).item()

    one_hot_labels = nn.functional.one_hot(
        labels, num_classes=10
    ).float()

    brier = torch.sum(
        (probabilities - one_hot_labels) ** 2,
        dim=1
    ).mean().item()

    ece = calculate_ece(probabilities, labels)

    avg_confidence = probabilities.max(dim=1)[0].mean().item()

    return {
        "Accuracy": accuracy,
        "NLL": nll,
        "Brier": brier,
        "ECE": ece,
        "Confidence": avg_confidence
    }


# --------------------------------
# 7. Compare Before and After
# --------------------------------

with torch.no_grad():

    scaled_eval_logits = eval_logits / temperature

    before = calculate_metrics(
        eval_logits, eval_labels
    )

    after = calculate_metrics(
        scaled_eval_logits, eval_labels
    )


print("\n========== CALIBRATION RESULTS ==========")

print(f"{'Metric':<15} {'Before':>12} {'After':>12}")

for metric in before:

    print(
        f"{metric:<15} "
        f"{before[metric]:>12.4f} "
        f"{after[metric]:>12.4f}"
    )


# --------------------------------
# 8. Save Temperature
# --------------------------------

torch.save(
    {
        "temperature": temperature
    },
    TEMPERATURE_PATH
)

print("\nTemperature saved:", TEMPERATURE_PATH)


# --------------------------------
# 9. Reliability Diagram
# --------------------------------

def reliability_values(probabilities, labels, bins=15):

    confidences, predictions = probabilities.max(dim=1)

    accuracies = predictions.eq(labels).float()

    bin_edges = torch.linspace(0, 1, bins + 1, device=DEVICE)

    bin_confidences = []
    bin_accuracies = []

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        if mask.any():

            bin_confidences.append(
                confidences[mask].mean().item()
            )

            bin_accuracies.append(
                accuracies[mask].mean().item()
            )

    return bin_confidences, bin_accuracies


with torch.no_grad():

    before_probs = torch.softmax(eval_logits, dim=1)

    after_probs = torch.softmax(
        scaled_eval_logits, dim=1
    )

    before_conf, before_acc = reliability_values(
        before_probs, eval_labels
    )

    after_conf, after_acc = reliability_values(
        after_probs, eval_labels
    )


plt.figure(figsize=(8, 7))

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.plot(
    before_conf, before_acc,
    marker="o",
    label="Before Calibration"
)

plt.plot(
    after_conf, after_acc,
    marker="s",
    label="After Calibration"
)

plt.xlabel("Average Confidence")
plt.ylabel("Actual Accuracy")
plt.title("Version 1 Reliability Diagram")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "v1_reliability_diagram.png"
)

plt.savefig(plot_path, dpi=300)
plt.show()

print("Reliability diagram saved:", plot_path)
print("\nCalibration completed.")
