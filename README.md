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
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">CIFAR10 and Eye Disease ResNet models</h3>

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
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
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
A classic deep residual network, implemented from the ground up and trained on the CIFAR-10 dataset. This model follows the architecture and training procedure described in the original ResNet paper, including data augmentation, multi-step learning rate scheduling, and weight decay.

2. Custom CNN on a Kaggle Open Dataset
A second model trained on an open-source dataset from [Kaggle](https://www.kaggle.com/datasets/ruhulaminsharif/eye-disease-image-dataset), showcasing flexibility in applying deep learning fundamentals to a different, real-world dataset. This part of the project involved adapting preprocessing steps and tuning hyperparameters based on the unique characteristics of the dataset.

📚 What I Learned:

- Building CNN architectures (including residual blocks) from scratch
- Dataset handling with torchvision and custom transformations
- Training & evaluation loops using DataLoader, loss functions, and optimizers
- Learning rate scheduling, TensorBoard logging, and model checkpointing
- Model performance analysis and iterative tuning

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- model results-->
## Results

Results are recorded from models trained locally on my setup:

### 🛠️ Training Setup

- GPU: AMD Navi 21 RX 6900XT
- PyTorch: PyTorch (2.6.0) ROCm 6.2.4
- OS: Ubuntu 24.04.2

### 📊 Final Model Performance

| Model         | Dataset      | Test Accuracy | Epochs | Parameters     | Training Time (s) |
|---------------|--------------|---------------|--------|----------------|-------------------|
| ResNet-20     | CIFAR-10     | 86.51%        | 64     | 0.27 Million   | 806.1353          |
| ResNet-20     | CIFAR-10     | 91.10%        | 128    | 0.27 Million   | 1607.8371         |

<!-- GETTING STARTED -->
## Run the project

Here is a guide to setting up the project locally.

### Prerequisites

Ensure you have the following:

- python (3.13.3)
- CUDA / ROCm installed if availiable (CPU training takes a very long time)

### Installation

1.
2. Clone the repo

   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```

3. Install NPM packages

   ```sh
   npm install
   ```

4. Enter your API in `config.js`

   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

5. Change git remote url to avoid accidental pushes to base project

   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

*For more examples, please refer to the [Documentation](https://example.com)*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
  - [ ] Chinese
  - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - <email@example.com>

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

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
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[Python.org]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch.org]:https://pytorch.org/
[PyTorch-url]:https://pytorch.org/
