## Oregon State University EECS Capstone 011: Enhancing Localized Deformation Analysis with AI/ML

# Crack Clustering Module

Crack-Clustering is an interactive toolkit for exploring crack patterns in SEM imagery. It slices or ingests whole JPEG images, extracts deep features with a ResNet-18 backbone, and automatically finds an optimal number of K-means clusters using elbow and silhouette analyses. Results are projected to 2-D with PCA and rendered in a Dash dashboard where each point links back to its full-resolution patch for rapid visual inspection. A single command launches the app, letting you switch between image-level and patch-level clustering, toggle green-outline preprocessing, and adjust patch size—all.

# Vector Field Generator Module
Vector-Field-Generator is a lightweight CLI utility that turns a single SEM micrograph—with a green-outlined crack—into an intuitive vector field that points away from the defect. The script segments the crack via HSV thresholding, fills the contour, computes a distance transform on the surrounding material, and derives unit-length gradients to visualize local “escape” directions. One command
```
python vector_field.py --image path/to/your_image.jpg
```
produces an annotated quiver plot, with adjustable arrow spacing and scale.

<!--

**Here are some ideas to get you started:**

🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->
