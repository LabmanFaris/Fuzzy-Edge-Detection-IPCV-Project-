This project implements an edge detection method for **MRI brain scans** using a **fuzzy logic–based rule system** (32 rules). The goal is to identify structural boundaries in brain MRI images more robustly than classical gradient-based methods.  
It compares the fuzzy-rule method with traditional edge detectors like **Sobel** and **Canny**, showing how fuzzy logic can produce clean and meaningful edges in medical imaging contexts.

Features
- Uses **Riddler’s thresholding** to binarize MRI images.  
- Applies a 32-rule fuzzy logic system over a 3×3 neighborhood to detect edges.  
- Organized rules by the number of bright neighbors for efficiency.  
- Comparison with Sobel and Canny edge detectors.  
- Supports different MRI orientations: axial, coronal, sagittal.  
- Implemented in Python using NumPy and OpenCV for image processing.
