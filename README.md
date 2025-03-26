# Regularization_Deep_learning
Jupyter Notebook demonstrating data augmentation techniques using Keras for image classification tasks. 
# Data Augmentation Using Keras

## Overview
This repository contains a **Jupyter Notebook** demonstrating **data augmentation** techniques using **Keras** for **image classification** tasks. Data augmentation is an essential technique in deep learning, allowing for improved model generalization by artificially expanding the dataset through transformations.

---

## Implementation Steps
1. **Import Required Libraries**: TensorFlow, Keras, NumPy, Matplotlib.
2. **Load Dataset**: Use a sample dataset or a custom image dataset.
3. **Define Data Augmentation Techniques**:  
   - Rotation  
   - Zooming  
   - Flipping  
   - Shearing  
   - Brightness Adjustment  
   - Rescaling  
4. **Use `ImageDataGenerator` from Keras** to apply augmentation.
5. **Visualize Augmented Images** before training.
6. **Train a CNN Model** with augmented images.
7. **Evaluate Model Performance** with and without augmentation.
8. **Make Predictions** and display results.

---

## Files in this Repository
- `Data_Augmentation_Using_Keras.ipynb` → Jupyter Notebook with code and explanations
- `datasets/` → Sample dataset folder for testing
- `README.md` → This documentation file

---

## How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Data_Augmentation_Using_Keras.git
   cd Data_Augmentation_Using_Keras
   ```
2. **Install dependencies**:
   ```bash
   pip install tensorflow keras numpy matplotlib
   ```
3. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Run the `Data_Augmentation_Using_Keras.ipynb` file**.

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV (optional for custom preprocessing)

