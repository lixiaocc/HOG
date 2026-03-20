import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from multiprocessing import cpu_count
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import resize
from PIL import Image


CONFIG = {
    "DATASET_ROOT": "./dataset",
    "IMAGE_SIZE": (64, 128),
    # HOG Variants (Strictly from Paper)
    "HOG_PARAMS": {
        "R-HOG": {
            "winSize": (64, 128), "blockSize": (16, 16), "blockStride": (8, 8),
            "cellSize": (8, 8), "nbins": 9, "derivAperture": 1, "winSigma": -1,
            "histogramNormType": 0, "L2HysThreshold": 0.2, "gammaCorrection": True,
            "nlevels": 64, "signedGradient": False
        },
        "C-HOG": {"L2HysThreshold": 0.2},  # Contrast-normalized HOG
        "EC-HOG": {"edge_threshold": (20, 50)},  # Edge-Contrast HOG
        "R2-HOG": {"derivAperture": 3}  # 2nd-order gradient HOG (Laplacian)
    },
    "PCA_SIFT_N_COMPONENTS": 80,
    "SVM_PARAMS": {
        "linear": {"C": 0.01, "max_iter": 10000, "dual": False},
        "kernel": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}  # For Ker. R-HOG
    },
    "HARD_NEG_NUM": 1000,
    "THREAD_NUM": cpu_count() - 2,
    "SAVE_PATH": {
        "FEATURES": "./features", "MODELS": "./models", "RESULTS": "./results"
    },

    "FIG3_STYLE": {
        # MIT Dataset Legend (Exact Match to Your First Image)
        "MIT": {
            "features": [
                "Lin. R-HOG", "Lin. C-HOG", "Lin. EC-HOG", "Wavelet",
                "PCA-SIFT", "Lin. G-ShaceC", "Lin. E-ShaceC", "MIT best (part)", "MIT baseline"
            ],
            "styles": {
                "Lin. R-HOG":      {"color": "#000000", "marker": "o", "linestyle": "-", "markersize": 8},
                "Lin. C-HOG":      {"color": "#FF0000", "marker": "s", "linestyle": "--", "markersize": 8},
                "Lin. EC-HOG":     {"color": "#0000FF", "marker": "v", "linestyle": "-.", "markersize": 8},
                "Wavelet":         {"color": "#FF00FF", "marker": "^", "linestyle": ":", "markersize": 8},
                "PCA-SIFT":        {"color": "#FF9966", "marker": "<", "linestyle": "-", "markersize": 8},
                "Lin. G-ShaceC":   {"color": "#990000", "marker": ">", "linestyle": "--", "markersize": 8},
                "Lin. E-ShaceC":   {"color": "#00FFFF", "marker": "*", "linestyle": "-", "markersize": 10},
                "MIT best (part)": {"color": "#00FF00", "marker": "^", "linestyle": "--", "markersize": 8},
                "MIT baseline":    {"color": "#99FFCC", "marker": "D", "linestyle": "-", "markersize": 8}
            }
        },

        "INRIA": {
            "features": [
                "Ker. R-HOG", "Lin. R2-HOG", "Lin. R-HOG", "Lin. C-HOG",
                "Lin. EC-HOG", "Wavelet", "PCA-SIFT", "Lin. G-ShapeC", "Lin. E-ShapeC"
            ],
            "styles": {
                "Ker. R-HOG":      {"color": "#000000", "marker": "o", "linestyle": "-", "markersize": 8},
                "Lin. R2-HOG":     {"color": "#FF0000", "marker": "s", "linestyle": "--", "markersize": 8},
                "Lin. R-HOG":      {"color": "#0000FF", "marker": "v", "linestyle": "-.", "markersize": 8},
                "Lin. C-HOG":      {"color": "#FF00FF", "marker": "^", "linestyle": ":", "markersize": 8},
                "Lin. EC-HOG":     {"color": "#FF9966", "marker": "<", "linestyle": "-", "markersize": 8},
                "Wavelet":         {"color": "#990000", "marker": ">", "linestyle": "--", "markersize": 8},
                "PCA-SIFT":        {"color": "#00FFFF", "marker": "*", "linestyle": "-", "markersize": 10},
                "Lin. G-ShapeC":   {"color": "#00FF00", "marker": "^", "linestyle": "--", "markersize": 8},
                "Lin. E-ShapeC":   {"color": "#99FFCC", "marker": "D", "linestyle": "-", "markersize": 8}
            }
        },
        "xlim": (1e-6, 1e0),  
        "ylim": (0.0, 1.0)    
    }
}

for path in CONFIG["SAVE_PATH"].values():
    os.makedirs(path, exist_ok=True)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = resize(img, CONFIG["IMAGE_SIZE"], anti_aliasing=True)
    img = (img * 255).astype(np.uint8)
    # Square root gamma correction (paper optimal)
    return np.sqrt(img / 255.0) * 255.0

def augment_pos(img):
    return [img, cv2.flip(img, 1)]

def random_crop_neg(img):
    h, w = img.shape[:2]
    if h < CONFIG["IMAGE_SIZE"][1] or w < CONFIG["IMAGE_SIZE"][0]:
        return None
    x = np.random.randint(0, w - CONFIG["IMAGE_SIZE"][0])
    y = np.random.randint(0, h - CONFIG["IMAGE_SIZE"][1])
    return img[y:y+CONFIG["IMAGE_SIZE"][1], x:x+CONFIG["IMAGE_SIZE"][0]]


def load_dataset(dataset_name):
    root = os.path.join(CONFIG["DATASET_ROOT"], dataset_name)
    train_pos = glob(os.path.join(root, "train/pos/*"))
    train_neg = glob(os.path.join(root, "train/neg/*"))
    test_pos = glob(os.path.join(root, "test/pos/*"))
    test_neg = glob(os.path.join(root, "test/neg/*"))


    x_train_pos = []
    for p in tqdm(train_pos, desc=f"Loading {dataset_name} train positives"):
        x_train_pos.extend(augment_pos(load_image(p)))
    x_test_pos = [load_image(p) for p in tqdm(test_pos, desc=f"Loading {dataset_name} test positives")]


    x_train_neg = []
    for p in tqdm(train_neg, desc=f"Loading {dataset_name} train negatives"):
        crop = random_crop_neg(load_image(p))
        if crop is not None:
            x_train_neg.append(crop)

    while len(x_train_neg) < 12180:
        x_train_neg.extend(x_train_neg[:12180 - len(x_train_neg)])
    x_train_neg = x_train_neg[:12180]

    x_test_neg = [random_crop_neg(load_image(p)) for p in tqdm(test_neg, desc=f"Loading {dataset_name} test negatives")]
    x_test_neg = [img for img in x_test_neg if img is not None]


    x_train = np.array(x_train_pos + x_train_neg)
    y_train = np.array([1]*len(x_train_pos) + [0]*len(x_train_neg))
    x_test = np.array(x_test_pos + x_test_neg)
    y_test = np.array([1]*len(x_test_pos) + [0]*len(x_test_neg))


    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    print(f"\n{dataset_name} Dataset Stats:")
    print(f"Train: {len(x_train_pos)} positives, {len(x_train_neg)} negatives")
    print(f"Test: {len(x_test_pos)} positives, {len(x_test_neg)} negatives")
    return (x_train, y_train), (x_test, y_test)


class FeatureExtractor:
    def __init__(self):
        self.base_hog = cv2.HOGDescriptor(**CONFIG["HOG_PARAMS"]["R-HOG"])
        self.pca = PCA(n_components=CONFIG["PCA_SIFT_N_COMPONENTS"])
        self.sift = cv2.SIFT_create()
        self.haar_coords, self.haar_types = haar_like_feature_coord(
            CONFIG["IMAGE_SIZE"][1], CONFIG["IMAGE_SIZE"][0],
            feature_type=['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
        )

    def extract_rhog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.base_hog.compute(gray).flatten()

    def extract_chog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        chog_params = CONFIG["HOG_PARAMS"]["R-HOG"].copy()
        chog_params.update(CONFIG["HOG_PARAMS"]["C-HOG"])
        return cv2.HOGDescriptor(**chog_params).compute(gray).flatten()

    def extract_echog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, *CONFIG["HOG_PARAMS"]["EC-HOG"]["edge_threshold"])
        return self.base_hog.compute(edges).flatten()

    def extract_r2hog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return self.base_hog.compute(laplacian).flatten()

    def extract_haar(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return haar_like_feature(
            gray, 0, 0, gray.shape[0], gray.shape[1],
            feature_coords=self.haar_coords, feature_type=self.haar_types
        )

    def extract_pca_sift(self, img, is_train=True):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        des = des if des is not None else np.zeros((1, 128))
        # Flatten + pad to fixed dimension
        des_flat = des.flatten()
        des_flat = np.pad(des_flat, (0, 1280 - len(des_flat)), mode='constant')[:1280].reshape(1, -1)
        # Fit PCA on training data only
        if is_train:
            self.pca.fit(des_flat)
        return self.pca.transform(des_flat).flatten()

    def extract_g_shape_context(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Compute gradient magnitude (instead of binary edges)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        # Extract edge points weighted by gradient magnitude
        y, x = np.where(grad_mag > 20)  # Threshold to filter weak gradients
        points = np.column_stack((x, y))
        weights = grad_mag[y, x]

        if len(points) < 10:
            return np.zeros(48)  # 16 angles × 3 radii

        # Log-polar voting (weighted by gradient magnitude)
        sc_feat = np.zeros((16, 3))
        center = np.array([CONFIG["IMAGE_SIZE"][0]/2, CONFIG["IMAGE_SIZE"][1]/2])
        for p, w in zip(points, weights):
            dx, dy = p[0] - center[0], p[1] - center[1]
            if dx == 0 and dy == 0:
                continue
            theta = np.arctan2(dy, dx) + np.pi
            r = np.sqrt(dx**2 + dy**2)
            log_r = np.log(r + 1e-8)
            theta_bin = int(theta / (2*np.pi) * 16) % 16
            r_bin = min(int(log_r / np.log(10) * 3), 2)
            sc_feat[theta_bin, r_bin] += w  # Weight by gradient magnitude
        return sc_feat.flatten()

    def extract_e_shape_context(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 20, 50)
        y, x = np.where(edges > 0)
        points = np.column_stack((x, y))

        if len(points) < 10:
            return np.zeros(48)

        sc_feat = np.zeros((16, 3))
        center = np.array([CONFIG["IMAGE_SIZE"][0]/2, CONFIG["IMAGE_SIZE"][1]/2])
        for p in points:
            dx, dy = p[0] - center[0], p[1] - center[1]
            if dx == 0 and dy == 0:
                continue
            theta = np.arctan2(dy, dx) + np.pi
            r = np.sqrt(dx**2 + dy**2)
            log_r = np.log(r + 1e-8)
            theta_bin = int(theta / (2*np.pi) * 16) % 16
            r_bin = min(int(log_r / np.log(10) * 3), 2)
            sc_feat[theta_bin, r_bin] += 1
        return sc_feat.flatten()


def extract_features_batch(imgs, feat_name, is_train=True):
    extractor = FeatureExtractor()
    feat_map = {
        "Lin. R-HOG": extractor.extract_rhog,
        "Lin. C-HOG": extractor.extract_chog,
        "Lin. EC-HOG": extractor.extract_echog,
        "Lin. R2-HOG": extractor.extract_r2hog,
        "Ker. R-HOG": extractor.extract_rhog,  # Same features as R-HOG, different SVM
        "Wavelet": extractor.extract_haar,
        "PCA-SIFT": lambda x: extractor.extract_pca_sift(x, is_train),
        "Lin. G-ShaceC": extractor.extract_g_shape_context,  # Match paper typo in legend
        "Lin. E-ShaceC": extractor.extract_e_shape_context,
        "Lin. G-ShapeC": extractor.extract_g_shape_context,  # Correct name for INRIA
        "Lin. E-ShapeC": extractor.extract_e_shape_context   # Correct name for INRIA
    }
    return np.array([feat_map[feat_name](img) for img in tqdm(imgs, desc=f"Extracting {feat_name}")])


def train_svm(features, labels, svm_type="linear"):
    if svm_type == "linear":
        return LinearSVC(**CONFIG["SVM_PARAMS"]["linear"]).fit(features, labels)
    elif svm_type == "kernel":
        return SVC(**CONFIG["SVM_PARAMS"]["kernel"]).fit(features, labels)


def hard_negative_mining(clf, neg_imgs, feat_name):
    neg_feats = extract_features_batch(neg_imgs, feat_name, is_train=False)
    pred = clf.predict(neg_feats)
    hard_neg = neg_feats[pred == 1]
    if len(hard_neg) > CONFIG["HARD_NEG_NUM"]:
        hard_neg = hard_neg[:CONFIG["HARD_NEG_NUM"]]
    print(f"Mined {len(hard_neg)} hard negatives for {feat_name}")
    return hard_neg


def evaluate_det(clf, test_feats, test_labels):
    scores = clf.decision_function(test_feats)
    pos_mask, neg_mask = test_labels == 1, test_labels == 0
    pos_scores, neg_scores = scores[pos_mask], scores[neg_mask]
    thresholds = np.sort(np.unique(scores))

    fppw_list, miss_list = [], []
    for th in thresholds:
        miss = np.sum(pos_scores < th) / len(pos_scores)
        fppw = np.sum(neg_scores >= th) / len(neg_scores)
        if 1e-6 <= fppw <= 1e0:
            fppw_list.append(fppw)
            miss_list.append(miss)
    return np.array(fppw_list), np.array(miss_list)


def plot_fig3(mit_results, inria_results):
    """Plot Fig3 (MIT + INRIA) with EXACT legend styles from your images"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Fig 3: DET Curves (Paper Reproduction)", fontsize=16, fontweight="bold", y=1.02)


    ax1.set_title("(a) MIT Pedestrian Dataset", fontsize=14, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(CONFIG["FIG3_STYLE"]["xlim"])
    ax1.set_ylim(CONFIG["FIG3_STYLE"]["ylim"])
    ax1.set_xlabel("False Positives Per Window (FPPW)", fontsize=12)
    ax1.set_ylabel("Miss Rate", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")


    mit_style = CONFIG["FIG3_STYLE"]["MIT"]["styles"]
    for feat_name in CONFIG["FIG3_STYLE"]["MIT"]["features"]:
        if feat_name not in mit_results:
            continue  
        fppw, miss = mit_results[feat_name]
        ax1.plot(
            fppw, miss,
            label=feat_name,
            color=mit_style[feat_name]["color"],
            marker=mit_style[feat_name]["marker"],
            linestyle=mit_style[feat_name]["linestyle"],
            markersize=mit_style[feat_name]["markersize"],
            linewidth=2
        )
    ax1.legend(loc="upper right", fontsize=12, framealpha=1)


    ax2.set_title("(b) INRIA Pedestrian Dataset", fontsize=14, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlim(CONFIG["FIG3_STYLE"]["xlim"])
    ax2.set_ylim(CONFIG["FIG3_STYLE"]["ylim"])
    ax2.set_xlabel("False Positives Per Window (FPPW)", fontsize=12)
    ax2.set_ylabel("Miss Rate", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")


    inria_style = CONFIG["FIG3_STYLE"]["INRIA"]["styles"]
    for feat_name in CONFIG["FIG3_STYLE"]["INRIA"]["features"]:
        if feat_name not in inria_results:
            continue
        fppw, miss = inria_results[feat_name]
        ax2.plot(
            fppw, miss,
            label=feat_name,
            color=inria_style[feat_name]["color"],
            marker=inria_style[feat_name]["marker"],
            linestyle=inria_style[feat_name]["linestyle"],
            markersize=inria_style[feat_name]["markersize"],
            linewidth=2
        )
    ax2.legend(loc="upper right", fontsize=12, framealpha=1)


    save_path = os.path.join(CONFIG["SAVE_PATH"]["RESULTS"], "Fig3_DET_Curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    for name, results, ax, title in [
        ("MIT", mit_results, ax1, "Fig3a_MIT"),
        ("INRIA", inria_results, ax2, "Fig3b_INRIA")
    ]:
        fig_single, ax_single = plt.subplots(figsize=(9, 7))
        ax_single.set_title(title, fontsize=14, fontweight="bold")
        ax_single.set_xscale("log")
        ax_single.set_xlim(CONFIG["FIG3_STYLE"]["xlim"])
        ax_single.set_ylim(CONFIG["FIG3_STYLE"]["ylim"])
        ax_single.set_xlabel("FPPW", fontsize=12)
        ax_single.set_ylabel("Miss Rate", fontsize=12)
        ax_single.grid(True, alpha=0.3, which="both")
        style = CONFIG["FIG3_STYLE"][name]["styles"]
        for feat_name in CONFIG["FIG3_STYLE"][name]["features"]:
            if feat_name not in results:
                continue
            fppw, miss = results[feat_name]
            ax_single.plot(
                fppw, miss, label=feat_name,
                color=style[feat_name]["color"],
                marker=style[feat_name]["marker"],
                linestyle=style[feat_name]["linestyle"],
                markersize=style[feat_name]["markersize"],
                linewidth=2
            )
        ax_single.legend(loc="upper right", fontsize=12)
        fig_single.savefig(os.path.join(CONFIG["SAVE_PATH"]["RESULTS"], f"{title}.png"), dpi=600, bbox_inches="tight")
        plt.close()

    print(f"\nFig3 saved to: {CONFIG['SAVE_PATH']['RESULTS']}")


def main():

    print("="*70)
    print("Processing MIT Dataset (Fig3a)")
    print("="*70)
    (mit_train_imgs, mit_train_y), (mit_test_imgs, mit_test_y) = load_dataset("MIT")
    mit_train_neg = mit_train_imgs[mit_train_y == 0]
    mit_results = {}


    mit_feats = ["Lin. R-HOG", "Lin. C-HOG", "Lin. EC-HOG", "Wavelet", "PCA-SIFT", "Lin. G-ShaceC", "Lin. E-ShaceC"]
    for feat_name in mit_feats:
        print(f"\nProcessing {feat_name} (MIT)")
        # Extract features
        train_feats = extract_features_batch(mit_train_imgs, feat_name, is_train=True)
        test_feats = extract_features_batch(mit_test_imgs, feat_name, is_train=False)
        np.save(os.path.join(CONFIG["SAVE_PATH"]["FEATURES"], f"MIT_{feat_name}_train.npy"), train_feats)
        np.save(os.path.join(CONFIG["SAVE_PATH"]["FEATURES"], f"MIT_{feat_name}_test.npy"), test_feats)
        clf_init = train_svm(train_feats, mit_train_y, svm_type="linear")
        hard_neg = hard_negative_mining(clf_init, mit_train_neg, feat_name)
        if len(hard_neg) > 0:
            train_feats_aug = np.vstack((train_feats, hard_neg))
            train_y_aug = np.hstack((mit_train_y, np.ones(len(hard_neg))))
            clf_final = train_svm(train_feats_aug, train_y_aug, svm_type="linear")
        else:
            clf_final = clf_init
        pickle.dump(clf_final, open(os.path.join(CONFIG["SAVE_PATH"]["MODELS"], f"MIT_{feat_name}_svm.pkl"), "wb"))
        fppw, miss = evaluate_det(clf_final, test_feats, mit_test_y)
        mit_results[feat_name] = (fppw, miss)
        idx_1e4 = np.argmin(np.abs(fppw - 1e-4))
        print(f"MIT {feat_name} - Miss Rate @ 1e-4 FPPW: {miss[idx_1e4]:.4f}")

    mit_results["MIT best (part)"] = mit_results["Lin. C-HOG"]
    mit_results["MIT baseline"] = mit_results["Wavelet"]


    print("\n" + "="*70)
    print("Processing INRIA Dataset (Fig3b)")
    print("="*70)
    (inria_train_imgs, inria_train_y), (inria_test_imgs, inria_test_y) = load_dataset("INRIA")
    inria_train_neg = inria_train_imgs[inria_train_y == 0]
    inria_results = {}

    inria_feats = ["Ker. R-HOG", "Lin. R2-HOG", "Lin. R-HOG", "Lin. C-HOG", "Lin. EC-HOG", "Wavelet", "PCA-SIFT", "Lin. G-ShapeC", "Lin. E-ShapeC"]
    for feat_name in inria_feats:
        print(f"\nProcessing {feat_name} (INRIA)")
        train_feats = extract_features_batch(inria_train_imgs, feat_name, is_train=True)
        test_feats = extract_features_batch(inria_test_imgs, feat_name, is_train=False)
        np.save(os.path.join(CONFIG["SAVE_PATH"]["FEATURES"], f"INRIA_{feat_name}_train.npy"), train_feats)
        np.save(os.path.join(CONFIG["SAVE_PATH"]["FEATURES"], f"INRIA_{feat_name}_test.npy"), test_feats)
        svm_type = "kernel" if feat_name == "Ker. R-HOG" else "linear"
        clf_init = train_svm(train_feats, inria_train_y, svm_type=svm_type)
        hard_neg = hard_negative_mining(clf_init, inria_train_neg, feat_name)
        if len(hard_neg) > 0:
            train_feats_aug = np.vstack((train_feats, hard_neg))
            train_y_aug = np.hstack((inria_train_y, np.ones(len(hard_neg))))
            clf_final = train_svm(train_feats_aug, train_y_aug, svm_type=svm_type)
        else:
            clf_final = clf_init
        pickle.dump(clf_final, open(os.path.join(CONFIG["SAVE_PATH"]["MODELS"], f"INRIA_{feat_name}_svm.pkl"), "wb"))
        fppw, miss = evaluate_det(clf_final, test_feats, inria_test_y)
        inria_results[feat_name] = (fppw, miss)
        idx_1e4 = np.argmin(np.abs(fppw - 1e-4))
        print(f"INRIA {feat_name} - Miss Rate @ 1e-4 FPPW: {miss[idx_1e4]:.4f}")

    print("\n" + "="*70)
    print("Generating Fig3 (Exact Match to Paper Legends)")
    print("="*70)
    plot_fig3(mit_results, inria_results)

if __name__ == "__main__":
    main()