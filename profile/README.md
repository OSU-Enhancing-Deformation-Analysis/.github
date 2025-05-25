# Enhancing Localized Deformation Analysis with AI/ML

*Oregon State University EECS Capstone 011*

Welcome to GitHub for our Capstone project, where we are using machine learning techniques to measure strain in materials. This space hosts multiple repositories that together comprise a comprehensive pipeline for analyzing microscopic material deformations from Scanning Electron Microscopy images.

## Project Overview

Digital Image Correlation (DIC) is a widely used tool for analyzing material deformation. However, DIC requires fine-tuned parameters and often struggles with the low-contrast, noisy nature of SEM imagery. Traditional methods frequently rely on adding speckle patterns to aid image tracking. We aim to apply a Deep Learning approach to the conventional DIC workflow to improve motion extraction in the presence of noise, eliminating the need for speckle patterns.

## Project Structure

### [Motion Vector Prediction Model](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model)

This is the core machine learning model training setup for our project. This model predicts 2D motion vector fields between two input images. The primary goal is to learn the transformation that maps an initial image to a subsequent image. This is achieved by training a U-Net-like convolutional neural network with self-attention mechanisms. A key feature of this project is its sophisticated synthetic data generation pipeline, which generates diverse training samples by programmatically creating and combining vector fields, and using various geometric and procedural shapes to mask these fields.

The model is trained using PyTorch, and experiment tracking is managed with Weights & Biases (Wandb).

### [Displacement to strain calculation](https://github.com/OSU-Enhancing-Deformation-Analysis/displaceToStrain)

Once our model predicts motion displacement fields, we use this module to derive strain maps, which show how the material stretches, compresses, or shears across its surface.

This tool applies a Virtual Strain Gauge (VSG) technique to transform motion fields into a new set of strain tensors (XX, YY, XY) at each point in the image. These values give researchers a more meaningful understanding of how internal stresses are distributed throughout the material under examination.

<!-- Both of these repos are private, if you make them public then uncomment this.

### Crack Clustering Module

Crack-Clustering is an interactive toolkit for exploring crack patterns in SEM imagery. It slices or ingests whole JPEG images, extracts deep features with a ResNet-18 backbone, and automatically finds an optimal number of K-means clusters using elbow and silhouette analyses. Results are projected to 2-D with PCA and rendered in a Dash dashboard where each point links back to its full-resolution patch for rapid visual inspection. A single command launches the app, letting you switch between image-level and patch-level clustering, toggle green-outline preprocessing, and adjust patch size—all.

### Vector Field Generator Module

Vector-Field-Generator is a lightweight CLI utility that turns a single SEM micrograph—with a green-outlined crack—into an intuitive vector field that points away from the defect. The script segments the crack via HSV thresholding, fills the contour, computes a distance transform on the surrounding material, and derives unit-length gradients to visualize local “escape” directions. One command

```
python vector_field.py --image path/to/your_image.jpg

```

produces an annotated quiver plot, with adjustable arrow spacing and scale.

-->

### [Model Output Preview](https://github.com/OSU-Enhancing-Deformation-Analysis/Model-Output-Preview)

This project provides a small interface to test our machine learning models while we train them. This standalone interface allows us to visualize image sequences, motion vectors, and strain fields together in a clean, structured layout. This tool simplifies result inspection and enables researchers to compare raw inputs, machine-predicted motion, and derived strain in one place.

### [Crack detection algorithm](https://github.com/OSU-Enhancing-Deformation-Analysis/crack_detection)

As a preprocessing step, we built a crack detection system using the OpenCV library. This algorithm identifies and segments visible cracks in SEM imagery to help filter out regions that may distort strain calculations. By focusing only on intact regions, we make the predicted motion fields and resulting strain maps more accurate and physically meaningful.

### [Enhancing Deformation Analysis UI](https://github.com/OSU-Enhancing-Deformation-Analysis/EnhancingDeformationAnalysisUI)

This is the culmination of our work put into a single application that can be used to do material analysis. The application includes pages for loading, titling, and processing Scanning Electron Microscopy images, allowing a user to easily filter and section images for use later in the pipeline. They can then use these images as inputs to our machine learning models in the program to find the displacement and the strain of the material. The application also includes pages for viewing important statistical information about the images to find more interesting results with the sequence.

---

## Technical Foundations

<img width="771" alt="image" src="https://github.com/user-attachments/assets/3c524b89-3059-4a0a-88de-fd9001b48b47" />

> Examples of random motion fields generated in our synthetic dataset.
> 

### The Model Architecture

<img width="1623" alt="image" src="https://github.com/user-attachments/assets/7c656b7d-d2a8-4690-ba5c-65f07c97b63f" />

> Our model architecture feeds grayscale images (left) transition through convolution and max pool stages to extract motion features. The self-attention stage correlates the motion across the image before decoding the motion using deconvolution and convolution stages to generate the per pixel X and Y displacement fields (right).
> 

This Deep Neural Network is specifically designed to estimate the motion displacement between a pair of SEM images.

The architecture takes inspiration from the well-established **U-Net structure**, widely used in image segmentation and pixel-wise tasks. Recognizing the challenges of SEM image noise and the need to capture distant or sudden motion correlations in the material, we enhanced the U-Net with a **self-attention mechanism** in the bottleneck layer to allow the model to find correlations between the most important features in the motion.

### Model Output

The model takes a pair of grayscale SEM images at different times. It outputs an X and Y displacement field that maps each pixel location in the first image to its corresponding location in the second image.

<img width="754" alt="image" src="https://github.com/user-attachments/assets/07a0cda8-61c4-4907-a222-4f0d564fe332" />

> Real image sequence inputs and motion output from the model next to the traditional Digital Image Correlation displacement. The motion shows the two sides of the material moving apart.
> 

### Synthetic Data

Without ground truth training data, we turn to synthetic data generation for training the model.

We start with a set of **17,000 motionless SEM material images**. Each training example includes an original SEM image and that same image with a known displacement applied. To generate the known displacement, we create a random combination of stacked motion fields and shape masks. This process is shown in Figure 3. Each motion field is selected from 14 functions types and modified by randomizing rotation, position, scale, and amplitude. The regions get masked by one of 19 shape masks, each also randomized in rotation, position, and scale. Finally, the combined **motion field is applied to the SEM image** and fed to the model as a pair. Each training example includes the original tile, the transformed tile, and the known displacement.

<img width="775" alt="image" src="https://github.com/user-attachments/assets/6484ebf3-aff5-4961-a78e-92d315fa4adf" />

> A final motion field is generated using synthetic motion fields combined at random; then a shape mask is applied to the motion fields to create the final motion field. The final motion field is applied to an SEM image to generate a known displacement of the image.
> 

<img width="635" alt="image" src="https://github.com/user-attachments/assets/b165fc81-e72b-4f3c-bdcc-d5b29b2761c2" />

### Training Example

<img width="714" alt="image" src="https://github.com/user-attachments/assets/9825977e-92c1-4066-b2a0-faebaf6f7342" />

## Getting Started: Try it for yourself.

We recommend you start by using our [Enhancing Deformation Analysis UI](https://github.com/OSU-Enhancing-Deformation-Analysis/EnhancingDeformationAnalysisUI) to try using our machine learning models and other systems in an existing interface.

If you want to try training your own machine learning models based on another set of images, go to the [Motion Vector Prediction Model](https://github.com/OSU-Enhancing-Deformation-Analysis/CNN-motion-model) repository to learn more about setting up your own project and using our training code.

## Acknowledgements

This project was made possible through the Oregon State University EECS Capstone program, with guidance from faculty who inspired this project and drove us to make some great progress.
