import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def display_images(images, titles,output_name:str, cmap='gray'): 
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)): 
        plt.subplot(1, len(images), i+1)
        if len(img.shape) == 2:  
            plt.imshow(img, cmap=cmap) 
        else: 
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        plt.title(title)
        plt.axis('off') 
    plt.savefig(f'output_images/{output_name}.png')

                           
def image_negative(img):
    return 255 - img

def gamma_correction(img, gamma=1.5):
    img_normalized = img / 255.0
    corrected = np.power(img_normalized, gamma)
    return np.uint8(corrected * 255)

def log_transform(img):
    img = img.astype(np.float32) + 1    
    max_val = np.max(img)
    if max_val <= 0:
        max_val = 1   
    c = 255 / np.log(1 + max_val)
    log_img = c * np.log(img)
    return np.clip(log_img, 0, 255).astype(np.uint8)

def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return np.uint8(stretched)

def histogram_equalization(img): 
    if len(img.shape) == 2:
        return cv2.equalizeHist(img) 
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0]) 
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def intensity_level_slicing(img, min_val=100, max_val=200):
    sliced = np.where((img >= min_val) & (img <= max_val), 255, 0)
    return np.uint8(sliced) 

def bit_plane_slicing(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bit_planes = [(img >> i) & 1 for i in range(8)]
    return [np.uint8(bp * 255) for bp in bit_planes]


# Load images
gray_img = cv2.imread('input_images/gray_img.png')
color_img = cv2.imread('input_images/RGB_img.png')   

# 1. Image Negative
neg_gray = image_negative(gray_img)
neg_color = image_negative(color_img)
display_images([gray_img, neg_gray], ['Original Grayscale', 'Negative'],'image_negative_gray')
display_images([color_img, neg_color], ['Original Color', 'Negative'],'image_negative_RGB')

# 2. Gamma Correction
gamma_gray = gamma_correction(gray_img, gamma=2.2)
gamma_color = gamma_correction(color_img, gamma=2.2)
display_images([gray_img, gamma_gray], ['Original Grayscale', 'Gamma Corrected'],'gamma_correction_gray')
display_images([color_img, gamma_color], ['Original Color', 'Gamma Corrected'],'gamma_correction_RGB')

# 3. Logarithmic
log_gray = log_transform(gray_img)
log_color = log_transform(color_img)
display_images([gray_img, log_gray], ['Original Grayscale', 'Log Transformation'],'log_transform_gray')
display_images([color_img, log_color], ['Original Color', 'Log Transformation'],'log_transform_RGB')

# 4. Contrast Stretching
contrast_gray = contrast_stretching(gray_img)
contrast_color = contrast_stretching(color_img) 
display_images([gray_img, contrast_gray], ['Original Grayscale', 'Contrast Stretched'],'contrast_stretching_gray')
display_images([color_img, contrast_color], ['Original Color', 'Contrast Stretched'],'contrast_stretching_RGB')

# 5. Histogram Equalization
hist_eq_gray = histogram_equalization(gray_img)
hist_eq_color = histogram_equalization(color_img) 
display_images([gray_img, hist_eq_gray], ['Original Grayscale', 'Histogram Equalized'],'histogram_equalization_gray')
display_images([color_img, hist_eq_color], ['Original Color', 'Histogram Equalized'],'histogram_equalization_RGB')

# 6. Intensity Level Slicing
slice_gray = intensity_level_slicing(gray_img)
slice_color = intensity_level_slicing(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY))
display_images([gray_img, slice_gray], ['Original Grayscale', 'Sliced'],'intensity_level_slicing_gray')
display_images([color_img, cv2.cvtColor(slice_color, cv2.COLOR_GRAY2BGR)], ['Original Color', 'Sliced (Grayscale)'],'intensity_level_slicing_RGB')

# 7. Bit Plane Slicing
bit_planes = bit_plane_slicing(gray_img)
bit_planes_RGB = bit_plane_slicing(gray_img)
display_images(bit_planes, [f'Bit Plane {i}' for i in range(8)],'bit_plane_slicing_gray')
display_images(bit_planes_RGB, [f'Bit Plane {i}' for i in range(8)],'bit_plane_slicing_RGB')