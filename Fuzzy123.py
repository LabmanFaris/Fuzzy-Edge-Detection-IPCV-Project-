import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def riddlers_threshold(gray_img, tol=0.5, max_iter=100):
    img_flat = gray_img.flatten().astype(np.float64)
    t1 = img_flat.mean()
    
    for iteration in range(max_iter):
        g1 = img_flat[img_flat <= t1]
        g2 = img_flat[img_flat > t1]
        if len(g1) == 0 or len(g2) == 0:
            break
        mean_g1 = g1.mean()
        mean_g2 = g2.mean()
        t_new = (mean_g1 + mean_g2) / 2.0

        if abs(t_new - t1) < tol:
            break
        
        t1 = t_new
    
    return t1

def create_binary_image(gray_img, threshold):
    binary_img = np.zeros_like(gray_img, dtype=np.uint8)
    binary_img[gray_img > threshold] = 1
    return binary_img

def check_rule(neighbors, pattern):
    #check in neighbours match one of the specified 32 patterns
    ones_indices, zeros_indices = pattern
    for idx in ones_indices:
        if neighbors[idx] != 1:
            return False
    for idx in zeros_indices:
        if neighbors[idx] != 0:
            return False
    return True

def apply_32_fuzzy_rules(binary_img):
    """
    32 fuzzy rules as described in the paper.
    The rules detect edge pixels (dark pixels with at least one bright neighbor)

    Mask:
    N1    N2    N3
    N8    x     N4
    N7    N6    N5
    """
    rows, cols = binary_img.shape
    edge_img = np.zeros_like(binary_img, dtype=np.uint8)
    
    # Define the 32 rules as (ones_indices, zeros_indices)
    # Index mapping: N1->0, N2->1, N3->2, N4->3, N5->4, N6->5, N7->6, N8->7
    rules = [
        #Single neighbor is 1, all others are 0
        ([0], [1, 2, 3, 4, 5, 6, 7]),  #1: N1=1
        ([1], [0, 2, 3, 4, 5, 6, 7]),  #2: N2=1
        ([2], [0, 1, 3, 4, 5, 6, 7]),  #3: N3=1
        ([3], [0, 1, 2, 4, 5, 6, 7]),  #4: N4=1
        ([4], [0, 1, 2, 3, 5, 6, 7]),  #5: N5=1
        ([5], [0, 1, 2, 3, 4, 6, 7]),  #6: N6=1
        ([6], [0, 1, 2, 3, 4, 5, 7]),  #7: N7=1
        ([7], [0, 1, 2, 3, 4, 5, 6]),  #8: N8=1
        
        # Two adjacent neighbors are 1
        ([0, 1], [2, 3, 4, 5, 6, 7]),  #9: N1&N2=1
        ([1, 2], [0, 3, 4, 5, 6, 7]),  #10: N2&N3=1
        ([2, 3], [0, 1, 4, 5, 6, 7]),  #11: N3&N4=1
        ([3, 4], [0, 1, 2, 5, 6, 7]),  #12: N4&N5=1
        ([4, 5], [0, 1, 2, 3, 6, 7]),  #13: N5&N6=1
        ([5, 6], [0, 1, 2, 3, 4, 7]),  #14: N6&N7=1
        ([6, 7], [0, 1, 2, 3, 4, 5]),  #15: N7&N8=1
        ([0, 7], [1, 2, 3, 4, 5, 6]),  #16: N1&N8=1
        
        #Three neighbors are 1
        ([2, 3, 4], [0, 1, 5, 6, 7]),  #17: N3&N4&N5=1
        ([0, 6, 7], [1, 2, 3, 4, 5]),  #18: N1&N8&N7=1
        ([0, 1, 2], [3, 4, 5, 6, 7]),  #19: N1&N2&N3=1
        ([4, 5, 6], [0, 1, 2, 3, 7]),  #20: N5&N6&N7=1
        
        # Four neighbors are 1
        ([2, 3, 4, 5], [0, 1, 6, 7]),  #21: N3&N4&N5&N6=1
        ([0, 1, 6, 7], [2, 3, 4, 5]),  #22: N1&N2&N7&N8=1
        ([0, 5, 6, 7], [1, 2, 3, 4]),  #23: N1&N6&N7&N8=1
        ([1, 2, 3, 4], [0, 5, 6, 7]),  #24: N2&N3&N4&N5=1
        ([3, 4, 5, 6], [0, 1, 2, 7]),  #25: N4&N5&N6&N7=1
        ([0, 1, 2, 7], [3, 4, 5, 6]),  #26: N1&N2&N3&N8=1
        ([0, 1, 2, 3], [4, 5, 6, 7]),  #27: N1&N2&N3&N4=1
        ([4, 5, 6, 7], [0, 1, 2, 3]),  #28: N5&N6&N7&N8=1
        
        # Five neighbors are 1
        ([0, 1, 5, 6, 7], [2, 3, 4]),  #29: N1&N2&N6&N7&N8=1
        ([1, 2, 3, 4, 5], [0, 6, 7]),  #30: N2&N3&N4&N5&N6=1
        ([3, 4, 5, 6, 7], [0, 1, 2]),  #31: N4&N5&N6&N7&N8=1
        ([0, 1, 2, 3, 7], [4, 5, 6]),  #32: N1&N2&N3&N4&N8=1
    ]
    padded_img = np.pad(binary_img, (1, 1), mode='constant', constant_values=0)

    # Scan through the image
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Check if current pixel is dark(0)
            if padded_img[i, j] == 0:
                neighbors = [
                    padded_img[i-1, j-1],  # top-left
                    padded_img[i-1, j],    # top
                    padded_img[i-1, j+1],  # top-right
                    padded_img[i, j+1],    # right
                    padded_img[i+1, j+1],  # bottom-right
                    padded_img[i+1, j],    # bottom
                    padded_img[i+1, j-1],  # bottom-left
                    padded_img[i, j-1],    # left
                ]
                
                # Count neighbors which have non-zero pixel values
                count = sum(neighbors)
                
                # check rules if atleast 1 neighbour is bright
                if count > 0:
                    # Group rules by count for efficiency (as per FKB in paper)
                    if count == 1 and any(check_rule(neighbors, rules[r]) for r in range(0, 8)):
                        edge_img[i-1, j-1] = 255
                    elif count == 2 and any(check_rule(neighbors, rules[r]) for r in range(8, 16)):
                        edge_img[i-1, j-1] = 255
                    elif count == 3 and any(check_rule(neighbors, rules[r]) for r in range(16, 20)):
                        edge_img[i-1, j-1] = 255
                    elif count == 4 and any(check_rule(neighbors, rules[r]) for r in range(20, 28)):
                        edge_img[i-1, j-1] = 255
                    elif count == 5 and any(check_rule(neighbors, rules[r]) for r in range(28, 32)):
                        edge_img[i-1, j-1] = 255
    
    return edge_img

def fuzzy_edge_detection_paper_method(gray_img):
    threshold = riddlers_threshold(gray_img)
    print(f"Computed threshold: {threshold:.2f}")

    binary_img = create_binary_image(gray_img, threshold)

    edge_img = apply_32_fuzzy_rules(binary_img)
    
    return edge_img, binary_img, threshold

def result_grid(image_paths, output_path="comparison_grid.png"):
    #To create a grid for the final results
    num_images = len(image_paths)
    results = []
    for idx, img_path in enumerate(image_paths):
        print(f"Processing {idx+1}/{num_images}: {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping...")
            continue
        max_dim = 512
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        #fuzzy edge detection operation
        edge_img, _, _ = fuzzy_edge_detection_paper_method(img)
        
        #Sobel operation
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges = np.uint8(255 * sobel_edges / sobel_edges.max())
        
        #Canny operation
        canny_edges = cv2.Canny(img, 50, 150)
        
        results.append({
            'name': img_path,
            'original': img,
            'fuzzy': edge_img,
            'canny': canny_edges,
            'sobel': sobel_edges
        })
    
    if not results:
        print("Error: No images were successfully processed")
        return
    
    # creating a grid to hold all results and compare
    rows = len(results)
    cols = 4  # For Original, Fuzzy, Canny, Sobel
    
    fig = plt.figure(figsize=(16, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.02, wspace=0.01)
    
    # Column titles
    titles = ['Original Image', 'Fuzzy Edge (32 Rules)', 'Canny Edge', 'Sobel Edge']
    
    for row_idx, result in enumerate(results):
        # Original
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax1.imshow(result['original'], cmap='gray')
        if row_idx == 0:
            ax1.set_title(titles[0], fontsize=12, fontweight='bold')
        ax1.set_ylabel(f"Image {row_idx+1}", fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Fuzzy Edge
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.imshow(result['fuzzy'], cmap='gray')
        if row_idx == 0:
            ax2.set_title(titles[1], fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Canny
        ax3 = fig.add_subplot(gs[row_idx, 2])
        ax3.imshow(result['canny'], cmap='gray')
        if row_idx == 0:
            ax3.set_title(titles[2], fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Sobel
        ax4 = fig.add_subplot(gs[row_idx, 3])
        ax4.imshow(result['sobel'], cmap='gray')
        if row_idx == 0:
            ax4.set_title(titles[3], fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    fig.suptitle('Edge Detection Methods on MRI Brain Scans',      
                 fontsize=16, fontweight='bold', y=0.995)       # title of grid
    plt.show()
    return results

# Main Function
if __name__ == "__main__":

    image_paths = ["t1.png", "t2.png", "mri_brain.png", "mri_brain1.png", "mri1.jpg"]
    results = result_grid(image_paths, output_path="mri_edge_comparison_table.png")
    single_img = cv2.imread("tumor1.jpg", cv2.IMREAD_GRAYSCALE)
    
    if single_img is not None:
        edge_img, binary_img, threshold = fuzzy_edge_detection_paper_method(single_img)
        #-#-#
        sobel_x = cv2.Sobel(single_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(single_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges = np.uint8(255 * sobel_edges / sobel_edges.max())
        #-#-#
        canny_edges = cv2.Canny(single_img, 50, 150)
