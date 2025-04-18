import argparse
import re
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy

from train_cifar import ResNet110, ResNet20, ResNet32, ResNet44, ResNet56

# --- Configuration ---
MODEL_BASE_DIR = Path("./models/RESNET_CIFAR10")
IMAGE_BASE_DIR = Path("./src/images")
DEFAULT_MODEL_FILENAME = "cifar10_ResNet110_156.pth"
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]  # Use the same std dev values as above
TARGET_SIZE = (32, 32)

models = {
    "ResNet20": ResNet20(),
    "ResNet32": ResNet32(),
    "ResNet44": ResNet44(),
    "ResNet56": ResNet56(),
    "ResNet110": ResNet110(),
}

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

preprocess_transform = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.LANCZOS
        ),
        transforms.ToTensor(),  # Converts PIL (H,W,C) [0,255] to Tensor (C,H,W) [0.0, 1.0]
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),  # Normalizes Tensor
    ]
)


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description="Process an image using a specified model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "-d",
        "--debug",
        help="Show extra debugging info",
        action="store_true",
    )

    p.add_argument(
        "-m",
        "--model",
        help=f"Choose which model filename to load from {MODEL_BASE_DIR}. Defaults to '{DEFAULT_MODEL_FILENAME}'.",
        default=DEFAULT_MODEL_FILENAME,
        metavar="MODEL_FILENAME",
    )

    p.add_argument(
        "filename",
        type=str,
        help=f"Image filename located within the '{IMAGE_BASE_DIR}' directory.",
        metavar="IMAGE_FILENAME",
    )

    return p.parse_args()


def load_model(path, full_model_path):
    """
    Loads a model from models/RESNET_CIFAR10

    Returns:
        Model
    """
    modelname = re.split("_", path)[1]
    model = models[modelname]
    model.load_state_dict((torch.load(f"{full_model_path}", weights_only=True)))
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Loads and preprocesses an image for CIFAR-10 model compatibility using Torchvision.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: Preprocessed image as a Tensor (C, H, W), normalized.
                      Returns None if image loading fails.
    """
    try:
        # 1. Load Image
        img = Image.open(image_path)

        # 2. Convert to RGB
        img = img.convert("RGB")

        # 3. Apply all transformations
        tensor = preprocess_transform(img)
        return tensor

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None


def evaluate_image():
    pass


if __name__ == "__main__":
    if sys.version_info < (3, 11, 0):
        sys.stderr.write("You need python 3.11 or later to run this\n")
        sys.exit(1)

    args = cmdline_args()
    debug = args.debug

    # --- Path Construction and Validation ---
    model_filename = args.model
    image_filename = args.filename

    # Construct the full paths using pathlib
    full_model_path = MODEL_BASE_DIR / model_filename
    full_image_path = IMAGE_BASE_DIR / image_filename

    # Validate model path
    if not full_model_path.is_file():
        sys.stderr.write(
            f"Error: Model file not found or is not a file: {full_model_path}\n"
        )
        sys.stderr.write(
            f"       Ensure '{model_filename}' exists in '{MODEL_BASE_DIR}'\n"
        )
        sys.exit(1)
    elif debug:
        print(f"Model found: {full_model_path}")

    # Validate image path
    if not full_image_path.is_file():
        sys.stderr.write(
            f"Error: Image file not found or is not a file: {full_image_path}\n"
        )
        sys.stderr.write(
            f"       Ensure '{image_filename}' exists in '{IMAGE_BASE_DIR}'\n"
        )
        sys.exit(1)
    elif debug:
        print(f"Image found: {full_image_path}")

    if debug:
        print("\nArguments parsed and paths validated successfully.")
        print(f"Using Model: {full_model_path}")
        print(f"Processing Image: {full_image_path}")

    # Example: Placeholder for loading model and image
    model = load_model(model_filename, full_model_path)
    tensor = preprocess_image(full_image_path)

    if tensor is not None:
        if debug:
            print("Image preprocessed successfully!")
            print(f"Output shape: {tensor.shape}")  # Should be torch.Size([3, 32, 32])
            print(f"Output dtype: {tensor.dtype}")  # Should be torch.float32
            print(f"Output min value: {tensor.min():.4f}")
            print(f"Output max value: {tensor.max():.4f}")

        # --- Next Steps (PyTorch) ---
        # Add batch dimension
        input_batch = tensor.unsqueeze(0)  # Shape: torch.Size([1, 3, 32, 32])
        if debug:
            print(f"PyTorch batch shape: {input_batch.shape}")

        with torch.no_grad():
            output = model(input_batch)
            print(classes[int(torch.max(output, 1)[1])])

    sys.exit(0)
