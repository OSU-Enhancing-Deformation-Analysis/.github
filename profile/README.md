## Oregon State University EECS Capstone 011: Enhancing Localized Deformation Analysis with AI/ML

These are the different repositories that comprise the parts of our project. These are just quick overviews, so check out each one to learn more. For more details on the overall goal of the project read below.

# Crack Clustering Module

Crack-Clustering is an interactive toolkit for exploring crack patterns in SEM imagery. It slices or ingests whole JPEG images, extracts deep features with a ResNet-18 backbone, and automatically finds an optimal number of K-means clusters using elbow and silhouette analyses. Results are projected to 2-D with PCA and rendered in a Dash dashboard where each point links back to its full-resolution patch for rapid visual inspection. A single command launches the app, letting you switch between image-level and patch-level clustering, toggle green-outline preprocessing, and adjust patch size—all.

# Vector Field Generator Module
Vector-Field-Generator is a lightweight CLI utility that turns a single SEM micrograph—with a green-outlined crack—into an intuitive vector field that points away from the defect. The script segments the crack via HSV thresholding, fills the contour, computes a distance transform on the surrounding material, and derives unit-length gradients to visualize local “escape” directions. One command
```
python vector_field.py --image path/to/your_image.jpg
```
produces an annotated quiver plot, with adjustable arrow spacing and scale.


---


### Motivation

- Digital Image Correlation tools are commonly used to analyze the strain in materials by using computer vision to track a sequence of images.
- Tracking the images requires heavily tuned parameters and can struggle with noisy or complex displacement fields.
- Scanning Electron Microscopy images are very noisy, low contrast, and some have smooth surfaces. This often requires speckle patterns to be added to the material to track effectively.
- We aim to apply a Deep Learning approach to the traditional DIC workflow to improve motion extraction in the presence of noise eliminating the need for speckle patterns.

### Innovation

- Our Deep Learning model is purpose built to extract motion displacement fields between SEM image pairs without requiring speckle patterns.
- A diverse synthetic dataset allows the model to learn on noise-less complex motion patterns without needing to gather real world motion data.
- The training data then uses real material SEM images to provide realistic noise, patterns, contrast, and material properties.
- The model learns to ignore the noise patterns and imaging artifacts in the sequences to generalize on just the motion extraction and clearly finds the motion without speckle patterns.

<img width="771" alt="image" src="https://github.com/user-attachments/assets/3c524b89-3059-4a0a-88de-fd9001b48b47" />

> Examples of random motion fields generated in our synthetic dataset.


### The Model Architecture

<img width="1623" alt="image" src="https://github.com/user-attachments/assets/7c656b7d-d2a8-4690-ba5c-65f07c97b63f" />

> Our model architecture feeds grayscale images (left) transition through convolution and max pool stages to extract motion features. The self-attention stage correlates the motion across the image before decoding the motion using deconvolution and convolution stages to generate the per pixel X and Y displacement fields (right).

This Deep Neural Network is specifically designed to estimate the motion displacement between a pair of SEM images. 

The architecture takes inspiration from the well-established **U-Net structure**, widely used in image segmentation and pixel-wise tasks. Recognizing the challenges of SEM image noise and the need to capture distant or sudden motion correlations in the material, we enhanced the U-Net with a **self-attention mechanism** in the bottleneck layer to allow the model to find correlations between the most important features in the motion.

### Model Output

The model takes a pair of grayscale SEM images at different times. It outputs an X and Y displacement field that maps each pixel location in the first image to its corresponding location in the second image. 

<img width="754" alt="image" src="https://github.com/user-attachments/assets/07a0cda8-61c4-4907-a222-4f0d564fe332" />

> Real image sequence inputs and motion output from the model next to the traditional Digital Image Correlation displacement. The motion shows the two sides of the material moving apart.


### Synthetic Data

Without ground truth training data, we turn to synthetic data generation for training the model.

We start with a set of **17,000 motionless SEM material images**. Each training example includes an original SEM image and that same image with a known displacement applied. To generate the known displacement, we create a random combination of stacked motion fields and shape masks. This process is shown in Figure 3. Each motion field is selected from 14 functions types and modified by randomizing rotation, position, scale, and amplitude. The regions get masked by one of 19 shape masks, each also randomized in rotation, position, and scale. Finally, the combined **motion field is applied to the SEM image** and fed to the model as a pair. Each training example includes the original tile, the transformed tile, and the known displacement.


<img width="775" alt="image" src="https://github.com/user-attachments/assets/6484ebf3-aff5-4961-a78e-92d315fa4adf" />

> A final motion field is generated using synthetic motion fields combined at random; then a shape mask is applied to the motion fields to create the final motion field. The final motion field is applied to an SEM image to generate a known displacement of the image.

<img width="635" alt="image" src="https://github.com/user-attachments/assets/b165fc81-e72b-4f3c-bdcc-d5b29b2761c2" />


### Training Example

<img width="714" alt="image" src="https://github.com/user-attachments/assets/9825977e-92c1-4066-b2a0-faebaf6f7342" />
