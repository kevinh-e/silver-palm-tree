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
  <h2 align="center">CIFAR10 and Lung Radiography ResNet models</h2>

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
A similar second model trained on the open-source dataset from [Kaggle][https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database], showcasing flexibility in applying deep learning fundamentals to a different, real-world dataset. This part of the project involved adapting preprocessing steps and tuning hyperparameters based on the unique characteristics of the dataset.

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
| ResNet-20     | CIFAR-10     | **86.51%**    | 64     | 0.27 Million   | 806.1353          |
| ResNet-20     | CIFAR-10     | **91.10%**    | 128    | 0.27 Million   | 1607.8371         |
| ResNet-44     | CIFAR-10     | **92.81%**    | 128    | 0.66 Million   | 3171.6705         |
| ResNet-110    | CIFAR-10     | **92.93%**    | 156    | 1.73 Million   | 9066.0921|

| Model         | Dataset      | Test Accuracy | Epochs | Parameters     | Training Time (s) |
|---------------|--------------|---------------|--------|----------------|-------------------|
| ResNet-20     | COVID        | 90.69%        | 128    | 0.66 Million   | 33780.2212        |

<!-- GETTING STARTED -->
## Usage

Here is a guide to setting up the project locally.

### Prerequisites

Ensure you have the following:

- python (3.13.3)
- CUDA / ROCm installed if availiable (CPU training takes a very long time)

### Installation

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

4. [Install PyTorch for your setup](<https://pytorch.org/get-started/locally/>)

Now you can either train the model yourself or use the pretrained ones.

### Train models locally

You can train the models on your own hardware and specify the epochs and model

#### Train on CIFAR10

   ```sh
   python3 src/cifar10/train_cifar.py modelname [epochs]
   ```

#### Train on COVID dataset

   ```sh
   python3 src/cifar10/train_covid.py modelname [epochs]
   ```

Trained models are stored in `./models/(RESNET_CIFAR10 | RESNET_COVID)`.

### Testing your own images

You can test your own images to see which output class the model outputs.

#### Label Reference

| **CIFAR-10**           | **COVID Radiography Dataset**                         |
|------------------------|--------------------------------------------------|
| Airplane               | COVID-19                                        |
| Automobile             | Lung Opacity|
| Bird                   | Normal|
| Cat                    | ViralPneumonia|
| Deer                   | |
| Dog                    | |
| Frog                   | |
| Horse                  | |
| Ship                   | |
| Truck                  | |

**Ensure your images are in `./src/images/`**
Run a CIFAR10 Model:

   ```sh
   python3 src/cifar10/test_cifar.py [modelname]
   ```

Run a COVID Model:

   ```sh
   python3 src/covid/test_covid.py [modelname]
   ```

<br>
The CIFAR10 model defaults to a ResNet110 model with 156 epochs.
The COVID model default to a ResNet20 model with 128 epochs (5 Residual Block layers).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Website - <https://kevinh.dev/>
<br>
Email - <contact@kevinh.dev>
<br>
Project Link: [https://github.com/kevinh-e/silver-palm-tree](https://github.com/kevinh-e/silver-palm-tree)
<br>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

-M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

[Paper link](https://ieeexplore.ieee.org/document/9144185)

<br>
-Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images.

[Paper link](https://doi.org/10.1016/j.compbiomed.2021.104319)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.org]: <https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54>
[Python-url]: <https://www.python.org/>
[PyTorch.org]:<https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white>
[PyTorch-url]:<https://pytorch.org/>
