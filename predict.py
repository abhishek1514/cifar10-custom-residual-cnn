
import os
import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from model_v2 import CIFAR10CNN_V2


# --------------------------------
# 1. Configuration
# --------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "./models/best_model_v2.pth"

TEMPERATURE_PATH = "./models/v2_temperature.pth"

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

TEMPERATURE = 1.0


# --------------------------------
# 2. Image Preprocessing
# --------------------------------

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        cifar_mean,
        cifar_std
    )

])


# --------------------------------
# 3. Load Version  Model
# --------------------------------

def load_model():

    global TEMPERATURE

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = CIFAR10CNN_V2().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("Model loaded successfully.")

    # Load Version 2 calibration temperature
    if os.path.exists(TEMPERATURE_PATH):

        calibration_data = torch.load(
            TEMPERATURE_PATH,
            map_location="cpu",
            weights_only=True
        )

        TEMPERATURE = max(
            0.05,
            float(calibration_data["temperature"])
        )

        print(
            f"Calibration temperature loaded: {TEMPERATURE:.4f}"
        )

    else:

        TEMPERATURE = 1.0

        print(
            "No Version 2 calibration found. Using T = 1.0"
        )

    return model


# --------------------------------
# 4. Open Input Image
# --------------------------------

def open_input_image(image_input):

    try:

        # Streamlit uploaded file or image-like object
        if isinstance(image_input, Image.Image):

            image = image_input.copy()

        elif hasattr(image_input, "seek"):

            image_input.seek(0)

            image = Image.open(image_input)

        # Local image path
        elif isinstance(image_input, (str, os.PathLike)):

            if not os.path.exists(image_input):

                raise FileNotFoundError(
                    f"Image not found: {image_input}"
                )

            image = Image.open(image_input)

        else:

            raise TypeError(
                "Input must be an image path, "
                "PIL Image, or uploaded image file."
            )

        return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as error:

        raise ValueError(
            f"Invalid or corrupted image: {error}"
        ) from error


# --------------------------------
# 5. Image Prediction
# --------------------------------

def predict_image(image_input, model):

    global TEMPERATURE

    # Open image
    image = open_input_image(image_input)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    model.eval()

    # Model inference
    with torch.inference_mode():

        outputs = model(input_tensor)

        # Apply temperature scaling
        probabilities = F.softmax(
            outputs / TEMPERATURE,
            dim=1
        )

        # Get top 3 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    # Highest confidence prediction
    predicted_index = top_indices[0][0].item()

    predicted_class = CLASSES[predicted_index]

    confidence = (
        top_probabilities[0][0].item() * 100
    )

    # Top 3 results
    top_3 = []

    for i in range(3):

        class_index = top_indices[0][i].item()

        class_confidence = (
            top_probabilities[0][i].item() * 100
        )

        top_3.append({

            "class": CLASSES[class_index],

            "confidence": class_confidence

        })

    # Terminal output
    print("\n========== PREDICTION RESULTS ==========")

    print("Predicted Class:", predicted_class)

    print(f"Confidence: {confidence:.2f}%")

    if confidence < 80:

        print(
            "Warning: Prediction confidence is below 80%."
        )

    else:

        print(
            "Prediction confidence is at least 80%."
        )

    print("\n========== TOP 3 PREDICTIONS ==========")

    for i, item in enumerate(top_3, start=1):

        print(
            f"{i}. {item['class']}: "
            f"{item['confidence']:.2f}%"
        )

    return {

        "predicted_class": predicted_class,

        "confidence": confidence,

        "top_3": top_3

    }


# --------------------------------
# 6. Main Execution
# --------------------------------

if __name__ == "__main__":

    model = load_model()

    # Replace with your actual image path
    IMAGE_PATH = "./test_image.jpg"

    result = predict_image(
        IMAGE_PATH,
        model
    )
