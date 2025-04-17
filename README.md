<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align"center">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/kevinh-e/silver-palm-tree?style=flat&color=gold">
    <img alt="License" src="https://img.shields.io/badge/Apache_2.0-License-blue?style=flat&logo=apache">
    <img alt="LinkedIn" src="https://img.shields.io/badge/%40kevinhedev-linkedin-blue?style=flat">
</div>
<!-- PROJECT LOGO -->
<div align="center">
  <h2 align="center">CIFAR10 and Eye Disease ResNet models</h2>

  <p align="center">
    Deep Learning self taught project including multiple CNN and Residual CNN models
    <a href="https://github.com/kevinh-e/silver-palm-tree/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#what-i-learned">What I learned</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Model Results</a>
      <ul>
        <li><a href="#training-setup">Training setup</a></li>
        <li><a href="#final-model-performance">Model performance</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Python][Python.org]][Python-url] [![PyTorch][PyTorch.org]][Pytorch-url]

This project is part of my self-taught deep learning journey, where I’ve built and trained convolutional neural networks (CNNs) from scratch using PyTorch. The goal was to understand the entire training pipeline—data preprocessing, model architecture, training loops, evaluation, and optimization—without relying on pre-trained models.

The final models are interpreted replicas of the famous [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) paper that won the 2015 ImageNet [ILSVRC](https://image-net.org/challenges/LSVRC/).

The project includes the implementation and training of two key models:

1. ResNet-20 for CIFAR-10
A classic deep residual network, implemented from the ground up and trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. This model follows the architecture and training procedure described in the original [ResNet](https://arxiv.org/abs/1512.03385) paper, including data augmentation, multi-step learning rate scheduling, and weight decay.

2. Custom CNN on a Kaggle Open Dataset
A second model trained on an open-source dataset from [Kaggle](https://www.kaggle.com/datasets/ruhulaminsharif/eye-disease-image-dataset), showcasing flexibility in applying deep learning fundamentals to a different, real-world dataset. This part of the project involved adapting preprocessing steps and tuning hyperparameters based on the unique characteristics of the dataset.

### What I Learned

- Building CNN architectures (including residual blocks) from scratch
- Dataset handling with torchvision and custom transformations
- Training & evaluation loops using DataLoader, loss functions, and optimizers
- Learning rate scheduling, TensorBoard logging, and model checkpointing
- Model performance analysis and iterative tuning

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- model results-->
## Results

Results are recorded from models trained locally on my setup:

### Training Setup

- GPU: AMD Navi 21 RX 6900XT
- PyTorch: PyTorch (2.6.0) ROCm 6.2.4
- OS: Ubuntu 24.04.2

### Final Model Performance

| Model         | Dataset      | Test Accuracy | Epochs | Parameters     | Training Time (s) |
|---------------|--------------|---------------|--------|----------------|-------------------|
| ResNet-20     | CIFAR-10     | 86.51%        | 64     | 0.27 Million   | 806.1353          |
| ResNet-20     | CIFAR-10     | 91.10%        | 128    | 0.27 Million   | 1607.8371         |
| ResNet-44     | CIFAR-10     | 92.81%        | 128    | 0.66 Million   | 3171.6705         |
| ResNet-56     | Eye Disease  | 91.10%        | 150    | 0.66 Million   | 1607.8371         |

<!-- GETTING STARTED -->
## Usage

Here is a guide to setting up the project locally.

### Prerequisites

Ensure you have the following:

- python (3.13.3)
- CUDA / ROCm installed if availiable (CPU training takes a very long time)

Badge### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/kevinh-e/silver-palm-tree.git
   cd silver-palm-tree

   ```

2. Setup Python venv

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install packages

   ```sh
   pip install -r requirements.txt
   ```

4. Install PyTorch for your setup:
-[PyTorch get started](https://pytorch.org/get-started/locally/)

Now you can either train the model yourself or use the pretrained ones.

#### Train models locally

You can train the models on your own hardware by specifying the output path for the model
Train the CIFAR10 Model:

   ```sh
   python3 src/train_cifar.py [modelname] [epochs] output
   ```

Train the Eye Disease Model:

   ```sh
   python3 src/train_eye.py [modelname] [epochs] output
   ```

### Testing your own images

You can test your own images to see which output class the model outputs.

#### Label Reference

| **CIFAR-10**           | **Eye Disease Dataset**                         |
|------------------------|--------------------------------------------------|
| Airplane               | Central Serous Chorioretinopathy - Color Fundus |
| Automobile             | Diabetic Retinopathy                            |
| Bird                   | Disc Edema                                      |
| Cat                    | Glaucoma                                        |
| Deer                   | Healthy                                         |
| Dog                    | Macular Scar                                    |
| Frog                   | Myopia                                          |
| Horse                  | Pterygium                                       |
| Ship                   | Retinal Detachment                              |
| Truck                  | Retinitis Pigmentosa                            |

Run the pretrained CIFAR10 Model (ResNet-44 [128]):

   ```sh
   python3 src/test_cifar.py [modelname]
   ```

Run the pretrained Eye Disease Model (ResNet-56 [150]):

   ```sh
   python3 src/test_eye.py [modelname]
   ```

You can test run the models you trained on your own images as long as they are in the project directory:

   ```sh
   python3 src/test_cifar.py -c modelpath imagepath
   python3 src/test_eye.py -c modelpath imagepath
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Website - <https://kevinh.dev/>
Email - <contact@kevinh.dev>
Project Link: [https://github.com/kevinh-e/silver-palm-tree](https://github.com/kevinh-e/silver-palm-tree)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Papers and resources

- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
- [Malven's Grid Cheatsheet](https://grid.malven.co/)
- [Img Shields](https://shields.io)
- [GitHub Pages](https://pages.github.com)
- [Font Awesome](https://fontawesome.com)
- [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.org]: <https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54>
[Python-url]: <https://www.python.org/>
[PyTorch.org]:<https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white>
[PyTorch-url]:<https://pytorch.org/>
