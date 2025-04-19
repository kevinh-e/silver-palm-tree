# covid_dataset.py

import os
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.exposure import equalize_hist


class CovidRadiographyDataset(Dataset):
    """
    Custom PyTorch Dataset for the COVID-19 Radiography Dataset.

    Loads images and their corresponding masks, applies augmentations
    (horizontal flip) and histogram equalization, and returns a 4-channel
    tensor (RGB + Mask).

    Dataset structure expected:
    root_dir/
    ├── Class1/
    │   ├── images/
    │   │   ├── img1.png
    │   │   └── ...
    │   └── masks/
    │       ├── img1.png  # Mask corresponding to img1.png
    │       └── ...
    ├── Class2/
    │   └── ...
    └── ...

    Args:
        root_dir (str): Path to the main dataset directory
                       (e.g., './data/COVID/COVID-19_Radiography_Dataset').
        apply_horizontal_flip (bool): Whether to apply random horizontal flipping.
        transform (callable, optional): Optional transform to be applied
                                        after basic processing but before
                                        combining channels. Defaults to ToTensor.
                                        Note: Custom transforms handle flip/histeq.
    """

    def __init__(self, root_dir, apply_horizontal_flip=True, target_size=(256, 256)):
        super().__init__()
        self.root = root_dir
        self.flip = apply_horizontal_flip
        self.t_size = target_size
        self.i_paths = []
        self.m_paths = []
        self.labels = []
        self.cns = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.cns)}

        print(f"Found classes: {self.cns}")

        # get images and masks
        for class_name in self.cns:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            i_dir = os.path.join(class_dir, "images")
            m_dir = os.path.join(class_dir, "masks")

            if not os.path.isdir(i_dir) or not os.path.isdir(m_dir):
                print(
                    f"Warning: Skipping class '{class_name}' - 'images' or 'masks' folder not found."
                )
                continue

            i_files = sorted(
                [f for f in os.listdir(i_dir) if f.lower().endswith(".png")]
            )

            for i_name in i_files:
                i_path = os.path.join(i_dir, i_name)
                # Assume mask has the same name in the masks folder
                m_path = os.path.join(m_dir, i_name)

                if os.path.exists(m_path):
                    self.i_paths.append(i_path)
                    self.m_paths.append(m_path)
                    self.labels.append(class_idx)
                else:
                    print(f"Warning: Mask not found for image {i_path}. Skipping.")

        if not self.i_paths:
            raise RuntimeError(
                f"No image/mask pairs found in {root_dir}. Check dataset structure."
            )

        # declare to tensor transforms
        self.to_tensor = transforms.ToTensor()
        self.resize_img = transforms.Resize(
            self.t_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.resize_mask = transforms.Resize(
            self.t_size, interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.i_paths)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: (tensor, label) where tensor is the 4-channel image
                   (RGB + Mask) and label is the class index.
        """
        img_path = self.i_paths[idx]
        mask_path = self.m_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # --- Apply Augmentations ---

            # 1. Random Horizontal Flip (applied consistently to image and mask)
            perform_flip = self.flip and random.random() < 0.5
            if perform_flip:
                img = ImageOps.mirror(img)
                mask = ImageOps.mirror(mask)

            # 2. Resize image and mask
            img = self.resize_img(img)
            mask = self.resize_mask(mask)

            # 3. Histogram Equalization
            img_np = np.array(img)
            img_eq_np = np.zeros_like(img_np)

            for i in range(3):
                img_eq_np[:, :, i] = equalize_hist(img_np[:, :, i]) * 255
            img_eq_np = img_eq_np.astype(np.uint8)  # Ensure correct dtype
            img = Image.fromarray(img_eq_np)

            # --- Convert to Tensor ---
            img_tensor = self.to_tensor(img)
            mask_tensor = self.to_tensor(mask)

            # --- Combine Channels ---
            combined_tensor = torch.cat((img_tensor, mask_tensor), dim=0)
            # Shape: [4, H, W]

            return combined_tensor, label

        except Exception as e:
            print(f"Error loading sample at index {idx}: {img_path}")
            print(f"Error details: {e}")
            raise e
