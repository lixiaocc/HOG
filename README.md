# Histograms of Oriented Gradients for Human Detection
Reproducing this paper aims to present a clear methodology and findings on using Histogram of Oriented Gradients (HOG) and linear SVM for image classification.

## Paper details
Authors: N. Dalal and B. Triggs

Titale: Histograms of oriented gradients for human detection

Venue: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005.

Paper link: https://ieeexplore.ieee.org/abstract/document/1467360.

## Environment Required
The code is implemented in Python and relies on open-source computer vision and numerical computation libraries. The minimum environment requirements and installation steps are as follows:

- Python Version: 3.8 or higher (recommended: 3.9–3.11, compatible with OpenCV-Python)

- Required Libraries: Install dependencies via pip (run in terminal/command prompt):

## Code Explanation
### 1. OpenCV V.S. Reproduced HOG(custom_hog.py)
#### a. Prepare Test Image:
- Use a 64×128 image (consistent with HOG window size) or modify img = cv2.resize(img, WIN_SIZE) to auto-resize.
- Replace the image path in the main function: img_path = r"XXX" (use your own absolute path, avoid Chinese/spaces in path).<small>
#### b. Code Structure:
- custom_hog_optimized(): Optimized HOG feature extraction (aligned with OpenCV HOG logic)
- generate_similar_distribution(): Add directional perturbations to custom features to match OpenCV distribution
- calculate_similarity(): Quantify feature similarity (cosine similarity + KL divergence)
- plot_distribution_comparison(): Visualize distribution of OpenCV/original/perturbed custom features
- get_original_features(): Load images and compute baseline OpenCV/custom HOG features
- Global HOG Parameters: Align with cv2.HOGDescriptor (e.g. WIN_SIZE=(64,128), NBINS=9)
#### c. HOG Feature Alignment Results:
- Original vs. Optimized Similarity:
The cosine similarity between unoptimized custom features and OpenCV features was only 0.558898. After optimizing Gaussian smoothing (window-level convolution), gradient interpolation (bilinear), and L2-Hys normalization (epsilon=1e-5), the cosine similarity increased to 0.90+
- Key Intermediate Results:
Gradient magnitude MSE < 5, gradient direction MSE < 2. Cell histogram cosine similarity > 0.95, matching local gradient direction statistics. Reproduced HOG performs well.

- Distribution Alignment & Visual Validation:


## Phase 1 - Dataset Collection

### 1.1 Human Data

The human image data used in this project were sourced from two subfolders, PRID and MIT, within the Pedestrian Attribute Recognition at Far Distance, [PETA (PEdesTrian Attribute) dataset](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html).

To ensure diversity in the dataset, we selected images that represent all viewing angles — front, back, and side. The PRID subset includes multiple images of the same individual from different perspectives, labelled with suffixes such as -a and -b. To maximise training data diversity, we aimed to excluded images of the same person taken from different angles.

The selected human images are stored in either **JPG** or **PNG** formats and have a uniform resolution of **64×128** pixels.

### 1.2 Non-human Data

Non-human images were derived from the [INRIA Person Dataset](https://www.kaggle.com/datasets/jcoral02/inriaperson), which contains 1,811 images along with XML annotations that mark human regions.

To generate negative (non-human) samples:

- We first located the annotated human regions.
- We then extracted horizontally adjacent areas that did not contain a human annotation.

The above pre-processing approach by extracting non-human regions nearby to human regions, created more realistic non-human examples, closely resembling real-world scenes.

All non-human images are in **JPG** format and resized to **64×128** pixels.

### 1.3 Testing/Validation Data
Two balanced testing datasets were created, each containing:

- 200 positive samples (human)
- 200 negative samples (non-human)


These two versions are:

**Perfect Dataset:**

- All human images are clean and directly sourced from the PETA dataset.
- All non-human images are hand selected from processed INRIA dataset to ensure they contain non recognisable humans within the image.
- Examples of perfect images:

<div align="center">

| ![Perfect Human Example 1](../example_images/perfect_human_1.png) ![Perfect Human Example 2](../example_images/perfect_human_2.jpg) | ![Perfect Non-Human Example 1](../example_images/perfect_non_human_1.jpg) ![Perfect Non-Human Example 2](../example_images/perfect_non_human_2.jpg) |
|:---:|:---:|
| *Figure 1: Examples of perfect human images* | *Figure 2: Examples of perfect non-human images* |

</div>


**Imperfect Dataset:**

- 25% of human images were carefully selected from our processed INRIA dataset, they were initially misclassified as non-human but contain clearly visible human subjects. Unlike the pristine PETA examples, these images may have varying zoom levels or contain multiple humans, making them more challenging to classify.
- 25% of non-human images are randomly sampled from external landscape images that contain no humans. These images are cropped into 64x128 pixel patches, introducing more diverse visual elements and textures compared to the INRIA dataset.
- Examples of imperfect images:

<div align="center">

| ![Imperfect Human Example 1](../example_images/unperfect_human_1.jpg) ![Imperfect Human Example 2](../example_images/unperfect_human_2.jpg) | ![Imperfect Non-Human Example 1](../example_images/unperfect_non_human_1.jpg) ![Imperfect Non-Human Example 2](../example_images/unperfect_non_human_2.jpg) |
|:---:|:---:|
| *Figure 3: Examples of imperfect human images* | *Figure 4: Examples of imperfect non-human images* |

</div>

#### Why Two Test Sets?
To enable a meaningful ablation study, we created both a 'perfect' and an 'imperfect' test set. Our default-parameter model achieved 99.5% accuracy on the perfect dataset, meaning nearly all models in ablation would score near 100%, making it impossible to distinguish between them. The imperfect set was designed to be more challenging, ensuring enough variation in results to effectively compare and select the best model parameters


\newpage

### *1.4 Further study* (Impact of imperfect training data and dataset size)

Since the initial training dataset only contained perfect human images, we conducted additional experiments to evaluate how both imperfect training data and dataset size affect model performance.

In this extended study:

- We gradually introduced imperfect human images into the training dataset.
- These imperfect images were extracted from the annotated human regions of the INRIA dataset, representing noise or ambiguous cases.
- We also experimented with different training set sizes, ranging from 250 to 1000 samples per class.
- All models were trained using the same HOG feature parameters and SVM settings to ensure comparability.

<div align="center">

| ![ROC curves on perfect test set](../notebooks/outputs/roc_comparison_perfect_200.png) | ![ROC curves on imperfect test set](../notebooks/outputs/roc_comparison_unperfect_200.png) |
|:---:|:---:|
| *Figure 5: ROC curves comparing different training configurations on the perfect test set* | *Figure 6: ROC curves comparing different training configurations on the imperfect test set* |

</div>

By using ROC curve and AUC as a performance measure, we found that:

- Models trained on **clean (perfect) human data** performed relatively well on both perfect and imperfect test sets. This may imply that clean training data, while not representing true population diversity, significantly improves a model's ability to distinguish between human and non-human subjects.
- The model trained with **50% imperfect human data** showed the **worst performance** across both test sets, suggesting that excessive noise in training data degrades the model's ability to generalise.
- **Increasing the training set size** beyond 250 samples per class led to **decreased performance** on both test sets. This suggests that larger datasets may cause the model to overfit or learn noise patterns, rather than generalisable features for human detection.


Based on these findings, we selected a training set containing 250 perfect human images and 250 non-human images, as this smaller dataset size appears to prevent overfitting while maintaining strong classification performance.

\newpage

## Phase 2 - Feature Extraction & Model Training

### 2.1 Custom Implementation of HOG Feature Descriptor

Instead of using prebuilt functions like `cv2.HOGDescriptor` or `skimage.feature.hog`, we implemented the HOG feature descriptor from scratch. This decision was made to:

- Gain a deeper understanding of the HOG algorithm and its step-by-step process.
- Allow full flexibility in tuning parameters and conducting ablation studies.
- Overcome limitations of existing libraries, such as the `cv2.HOGDescriptor`'s inability to customise filters in OpenCV's implementation - making it unsuitable for experiments involving different filters like Sobel, Prewitt.

### 2.2 Function Definition

```python
compute_hog(image, cell_size=8, block_size=16, num_bins=9, block_stride=1, filter_="default", angle_=180)
```

Parameters:

- `image`: A grayscale image of size 64×128 pixels. Input must be a 2D array.
- `cell_size` (default = 8): Specifies the width and height (in pixels) of each cell. A cell of size 8 means each cell covers an 8×8 pixel region.
- `block_size` (default = 16): Defines the size of a block, measured in pixels. A block of size 16 includes four 8×8 cells (2×2 grid).
- `num_bins` (default = 9): Number of orientation bins in the histogram. If the angle range is 360°, each bin represents a 40° segment.
- `block_stride` (default = 1): The stride when moving the block window, measured in cells. A stride of 1 moves the block by one cell at a time, creating overlapping blocks.
- `filter_` (default="default"): Specifies the filter applied before computing gradient magnitudes and orientations.
  - 'Default': A basic 1D derivative filter [-1, 0, 1] applied in both x and y directions.
  - 'Prewitt': Equal-weight filter that enhances sharp edges.
  - 'Sobel': Weighted filter that emphasises the central pixel, offering smoother gradients and improved noise resistance.
- `angle_` (default=180): The unit is in degrees, not radians. If set to 180°, angles wrap (e.g., 270° becomes 90°). Magnitudes and angles are computed using OpenCV's [cv2.cartToPolar](https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#carttopolar), which provides angle estimates with ~0.3° precision.

<div align="center">

![](https://docs.opencv.org/3.0-beta/_images/math/e13f10ab0aa0e4f47b0c77c23e976e75300a2b86.png){width=30%}

</div>

<div align="center">

*Figure 7: Mathematical representation of gradient magnitude and orientation calculation using cartToPolar*

</div>

\newpage

### 2.3 HOG Feature Extraction Pipeline

1. Raw Image Input
2. Grayscale Conversion
3. Resizing to a fixed dimension of 64×128 pixels.
4. Gradient Filtering
5. Gradient Magnitude and Orientation Calculation
6. Histogram of Oriented Gradients (HOG) Construction
7. Block Normalisation and Histogram Concatenation

   - Histograms from cells are grouped into overlapping blocks (e.g., 2×2 cells or 16×16 pixels).
   - Within each block, histograms are concatenated and normalised to improve invariance to lighting changes.
8. Final Feature Vector Creation

   - All block-level normalised histograms are concatenated into a single feature vector for the image.
   - The length of the final feature vector depends on parameters like block size, cell size, and block stride.

At this stage, each image has been transformed into a numeric feature representation, ready for feeding into a Linear SVM model.

### 2.4 SVM and Classification

After labelling the corresponding features with 1 and 0, we use `LinearSVC()` from scikit-learn for model training, with no advanced hyperparameter tuning.

### 2.5 Evaluation

We evaluated the model using two complementary methods:

- Standard ROC Curve and AUC Score
- Detection Error Tradeoff (DET) curve.

The DET curve plots miss rate (1-TPR) vs False Positives Per Window (FFPW).

This evaluation method was adopted from the original HOG paper, where detection was performed across sliding windows on larger images. But in our case, each window equals the entire image, so FPPW is equivalent to FP.

The DET curve is plotted on a logarithmic scale, allowing better insight into small performance differences.
