This project implements an edge detection method for **MRI brain scans** using a **fuzzy logic–based rule system** (32 rules). The goal is to identify structural boundaries in brain MRI images more robustly than classical gradient-based methods.  
It compares the fuzzy-rule method with traditional edge detectors like **Sobel** and **Canny**, showing how fuzzy logic can produce clean and meaningful edges in medical imaging contexts.

Features
- Uses **Riddler’s thresholding** to binarize MRI images.  
- Applies a 32-rule fuzzy logic system over a 3×3 neighborhood to detect edges.  
- Organized rules by the number of bright neighbors for efficiency.  
- Comparison with Sobel and Canny edge detectors.  
- Supports different MRI orientations: axial, coronal, sagittal.  
- Implemented in Python using NumPy and OpenCV for image processing.


The following are the images upon which the various edge detection algorithms have been implemented.
<img width="1539" height="810" alt="results2" src="https://github.com/user-attachments/assets/3f183213-baff-41f2-9b48-a0f6db08b452" />
