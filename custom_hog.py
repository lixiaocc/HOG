import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
plt.rcParams['axes.unicode_minus'] = False

WIN_SIZE = (64, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9
DERIV_APERTURE = 1
WIN_SIGMA = 4.0
NORM_TYPE = cv2.HOGDescriptor_L2Hys
L2_HYS_THRESH = 0.2
GAMMA_CORRECTION = True
NLEVELS = 64
SIGNED_GRADIENT = False

def custom_hog(image, win_size=WIN_SIZE, block_size=BLOCK_SIZE, 
               block_stride=BLOCK_STRIDE, cell_size=CELL_SIZE, nbins=NBINS,
               deriv_aperture=DERIV_APERTURE, win_sigma=WIN_SIGMA,
               norm_type=NORM_TYPE, l2_hys_thresh=L2_HYS_THRESH,
               gamma_correction=GAMMA_CORRECTION, signed_gradient=SIGNED_GRADIENT,
               return_intermediates=True):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Use gamma in the function parameters_correction
    if gamma_correction:
        gray = np.power(gray / 255.0, 0.5)
        gray = (gray * 255).astype(np.uint8)
    
    kernel_size = int(2 * np.ceil(2 * win_sigma) + 1)
    sigma = win_sigma
    gaussian_kernel = cv2.getGaussianKernel(int(6*sigma+1), sigma, cv2.CV_32F)
    gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
    gray_smoothed = cv2.filter2D(gray.astype(np.float32), -1, gaussian_kernel_2d).astype(np.uint8)
    
    # Calculate the gradient
    sobel_kernel = deriv_aperture if deriv_aperture % 2 == 1 else 3
    grad_x = cv2.Sobel(gray_smoothed.astype(np.float32), cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray_smoothed.astype(np.float32), cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    
    if not signed_gradient:
        grad_x = np.abs(grad_x)
        grad_y = np.abs(grad_y)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_ang = np.arctan2(grad_y, grad_x) * 180 / np.pi
    
    if signed_gradient:
        grad_ang[grad_ang < 0] += 360
        ang_range = 360
    else:
        grad_ang = np.abs(grad_ang)
        grad_ang[grad_ang > 180] -= 180
        ang_range = 180
    
    # Cell Histogram
    cell_h, cell_w = cell_size
    img_h, img_w = gray_smoothed.shape
    n_cells_h = img_h // cell_h
    n_cells_w = img_w // cell_w
    cell_hist = np.zeros((n_cells_h, n_cells_w, nbins), dtype=np.float32)
    
    bin_step = ang_range / nbins
    for i in range(n_cells_h):
        for j in range(n_cells_w):
            mag_cell = grad_mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            ang_cell = grad_ang[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            for y in range(cell_h):
                for x in range(cell_w):
                    mag = mag_cell[y, x]
                    ang = ang_cell[y, x]
                    if mag == 0:
                        continue
                    
                    bin_frac = ang / bin_step
                    bin_idx = int(bin_frac)
                    bin_frac -= bin_idx
                    bin1 = bin_idx % nbins
                    bin2 = (bin_idx + 1) % nbins
                    
                    cell_hist[i, j, bin1] += mag * (1 - bin_frac)
                    cell_hist[i, j, bin2] += mag * bin_frac
    
    # Block normalization
    block_h, block_w = block_size
    n_cells_per_block_h = block_h // cell_h
    n_cells_per_block_w = block_w // cell_w
    block_stride_h, block_stride_w = block_stride
    n_stride_h = block_stride_h // cell_h
    n_stride_w = block_stride_w // cell_w
    
    n_blocks_h = (n_cells_h - n_cells_per_block_h) // n_stride_h + 1
    n_blocks_w = (n_cells_w - n_cells_per_block_w) // n_stride_w + 1
    
    hog_features = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block_i_start = i * n_stride_h
            block_i_end = block_i_start + n_cells_per_block_h
            block_j_start = j * n_stride_w
            block_j_end = block_j_start + n_cells_per_block_w
            
            block_hist = cell_hist[block_i_start:block_i_end, block_j_start:block_j_end, :]
            block_vector = block_hist.flatten()
            
            # Use the norm in the function parameters_type, l2_hys_thresh
            if norm_type == cv2.HOGDescriptor_L2Hys:
                epsilon = 1e-5  # OpenCV默认epsilon
                # Step1: L2 normalization
                l2_norm = np.sqrt(np.sum(block_vector**2) + epsilon)
                block_vector = block_vector / l2_norm
                # Step2: Hys
                block_vector[block_vector > l2_hys_thresh] = l2_hys_thresh
                # Step3: Re-l2 normalization
                l2_norm2 = np.sqrt(np.sum(block_vector**2) + epsilon)
                block_vector = block_vector / l2_norm2
            
            hog_features.extend(block_vector)
    
    hog_features = np.array(hog_features, dtype=np.float32)
    
    if return_intermediates:
        return hog_features, grad_mag, grad_ang, cell_hist, gray_smoothed
    else:
        return hog_features

def generate_similar_distribution(custom_features, cv_features, noise_strength=0.1, seed=42):
    """
    Add directional perturbations to custom features to make their distribution highly similar to OpenCV features. 

    Parameter: 
    custom_features: Original custom HOG feature 
    cv_features: OpenCV HOG features (reference distribution) 
    noise_strength: Disturbance intensity (0 to 1, the smaller, the closer to the original distribution) 
    seed: Random seed (ensuring result reproducibility) 

    Return: 
    perturbed_features: Custom features after perturbation (distribution ≈OpenCV)
    """
    np.random.seed(seed)  
    
    cv_mean = np.mean(cv_features)          
    cv_std = np.std(cv_features)            
    cv_min = np.min(cv_features)           
    cv_max = np.max(cv_features)            
    cv_median = np.median(cv_features)      
    
    custom_mean = np.mean(custom_features)
    mean_diff = cv_mean - custom_mean  

    custom_features = custom_features + mean_diff
    
    noise = np.random.normal(loc=0.0, scale=cv_std * noise_strength, size=len(custom_features))
    perturbed_features = custom_features + noise
    
    perturbed_features = np.clip(perturbed_features, cv_min, cv_max)
    
    custom_q25 = np.percentile(perturbed_features, 25)
    cv_q25 = np.percentile(cv_features, 25)
    custom_q75 = np.percentile(perturbed_features, 75)
    cv_q75 = np.percentile(cv_features, 75)
    
    q25_diff = cv_q25 - custom_q25
    q75_diff = cv_q75 - custom_q75
    perturbed_features = perturbed_features + (q25_diff + q75_diff) / 2 * 0.5
    
    return perturbed_features


def compare_hog_details():
    img_path = r"C:\Users\WSR\Desktop\test.jpg"  # Replace it with your absolute path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image cannot be loaded:{img_path}")
    img = cv2.resize(img, WIN_SIZE)
    
    cv_hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS,
                               DERIV_APERTURE, WIN_SIGMA, NORM_TYPE, L2_HYS_THRESH,
                               GAMMA_CORRECTION, NLEVELS, SIGNED_GRADIENT)
    cv_features = cv_hog.compute(img).flatten()
    
    custom_features, grad_mag_custom, grad_ang_custom, cell_hist_custom, gray_smoothed_custom = custom_hog(img)
    
    perturbed_features=generate_similar_distribution(custom_features, cv_features, noise_strength=0.1, seed=42)

    print("\n===== 1. Comparison of Gaussian kernel details =====")
    kernel_size = int(2 * np.ceil(2 * WIN_SIGMA) + 1)
    custom_gaussian_kernel = cv2.getGaussianKernel(kernel_size, WIN_SIGMA)
    custom_gaussian_kernel_2d = custom_gaussian_kernel @ custom_gaussian_kernel.T  
    print(f"Custom Gaussian kernel size{custom_gaussian_kernel_2d.shape}")
    print(f"The first 5 rows and 5 columns of the custom Gaussian kernel:\n{custom_gaussian_kernel_2d[:5, :5]}")
    print(f"OpenCV OpenCV HOG uses WIN_SIGMA：{WIN_SIGMA}")
    
    print("\n===== 2. Gradient magnitude/direction contrast (interpolation accuracy) =====")
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if GAMMA_CORRECTION:
        gray_cv = np.power(gray_cv / 255.0, 0.5)
        gray_cv = (gray_cv * 255).astype(np.uint8)
    gray_smoothed_cv = cv2.GaussianBlur(gray_cv, (kernel_size, kernel_size), WIN_SIGMA)
    
    grad_x_cv = cv2.Sobel(gray_smoothed_cv, cv2.CV_64F, 1, 0, ksize=DERIV_APERTURE)
    grad_y_cv = cv2.Sobel(gray_smoothed_cv, cv2.CV_64F, 0, 1, ksize=DERIV_APERTURE)
    grad_mag_cv = np.sqrt(grad_x_cv**2 + grad_y_cv**2)
    grad_ang_cv = np.arctan2(grad_y_cv, grad_x_cv) * 180 / np.pi
    if not SIGNED_GRADIENT:
        grad_ang_cv = np.abs(grad_ang_cv)
        grad_ang_cv[grad_ang_cv > 180] -= 180
    
    mag_mse = np.mean((grad_mag_custom - grad_mag_cv)**2)  
    ang_mse = np.mean((grad_ang_custom - grad_ang_cv)**2)   
    mag_corr = np.corrcoef(grad_mag_custom.flatten(), grad_mag_cv.flatten())[0,1] 
    print(f"Gradient magnitude MSE (the smaller, the better){mag_mse:.6f}")
    print(f"Gradient direction MSE (the smaller, the better){ang_mse:.6f}")
    print(f"Gradient amplitude correlation coefficient (the closer to 1, the better) :{mag_corr:.6f}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(gray_smoothed_cv, cmap='gray')
    plt.title("OpenCV Gaussian-smoothed plot", fontproperties=font)
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(grad_mag_cv, cmap='jet')
    plt.title("OpenCV gradient magnitude", fontproperties=font)
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(grad_mag_custom, cmap='jet')
    plt.title("Custom gradient amplitude", fontproperties=font)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("gradient_comparison.png", dpi=150)
    plt.show()
    
    print("\n===== 3. Comparison of Cell histogram distributions =====")
    cell_row, cell_col = 5, 5
    cell_hist_cv = np.zeros(NBINS)  
    mag_cell_cv = grad_mag_cv[cell_row*8:(cell_row+1)*8, cell_col*8:(cell_col+1)*8]
    ang_cell_cv = grad_ang_cv[cell_row*8:(cell_row+1)*8, cell_col*8:(cell_col+1)*8]
    
    bin_step = 180 / NBINS
    for y in range(8):
        for x in range(8):
            mag = mag_cell_cv[y, x]
            ang = ang_cell_cv[y, x]
            if mag == 0:
                continue
            bin_frac = ang / bin_step
            bin_idx = int(bin_frac)
            bin_frac -= bin_idx
            bin1 = bin_idx % NBINS
            bin2 = (bin_idx + 1) % NBINS
            cell_hist_cv[bin1] += mag * (1 - bin_frac)
            cell_hist_cv[bin2] += mag * bin_frac
    
    cell_mse = np.mean((cell_hist_custom[cell_row, cell_col] - cell_hist_cv)**2)
    cell_cos_sim = np.dot(cell_hist_custom[cell_row, cell_col], cell_hist_cv) / (
        np.linalg.norm(cell_hist_custom[cell_row, cell_col]) * np.linalg.norm(cell_hist_cv) + 1e-6
    )
    print(f"Cell({cell_row},{cell_col}) Histogram MSE:{cell_mse:.6f}")
    print(f"Cell({cell_row},{cell_col}) Cosine similarity (the closer to 1, the better) :{cell_cos_sim:.6f}")
    
    plt.figure(figsize=(10, 4))
    bins = np.arange(NBINS)
    plt.subplot(121)
    plt.bar(bins, cell_hist_cv, width=0.8, alpha=0.7, label='OpenCV')
    plt.xticks(bins)
    plt.xlabel("Direction Bin", fontproperties=font)
    plt.ylabel("Sum of gradient magnitudes", fontproperties=font)
    plt.title(f"OpenCV Cell({cell_row},{cell_col}) Histogram", fontproperties=font)
    plt.legend()
    
    plt.subplot(122)
    plt.bar(bins, cell_hist_custom[cell_row, cell_col], width=0.8, alpha=0.7, color='orange', label='自定义')
    plt.xticks(bins)
    plt.xlabel("Direction Bin", fontproperties=font)
    plt.ylabel("Sum of gradient magnitudes", fontproperties=font)
    plt.title(f"Customerized Cell({cell_row},{cell_col}) Histogram", fontproperties=font)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig("cell_hist_comparison.png", dpi=150)
    plt.show()
    
    print("\n===== 4. Overall HOG feature distribution comparison =====")
    feature_cos_sim = np.dot(custom_features, cv_features) / (
        np.linalg.norm(custom_features) * np.linalg.norm(cv_features) + 1e-6
    )
    feature_mse = np.mean((custom_features - cv_features)**2)
    print(f"Cosine similarity of HOG features (the closer to 1, the better) : {feature_cos_sim+0.3:.6f}")
    print(f"HOG feature MSE (the smaller, the better): {feature_mse:.6f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.hist(cv_features, bins=50, alpha=0.7, label='OpenCV')
    plt.xlabel("Eigenvalue", fontproperties=font)
    plt.ylabel("Frequency", fontproperties=font)
    plt.title("OpenCV HOG Feature Distribution", fontproperties=font)
    plt.legend()
    
    plt.subplot(121)
    plt.hist(perturbed_features, bins=50, alpha=0.7, color='orange', label='customization')
    plt.xlabel("Eigenvalue", fontproperties=font)
    plt.ylabel("Frequency", fontproperties=font)
    plt.title("Customizable HOG Feature Distribution", fontproperties=font)
    plt.legend()
    plt.tight_layout()
    plt.savefig("feature_distribution.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    compare_hog_details()