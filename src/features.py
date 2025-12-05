# src/features.py
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------- Text features ----------
class TextFeatureExtractor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

# ---------- Image features ----------
def color_histogram(image_path, bins=(8,8,8)):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(np.prod(bins), dtype=float)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256]).flatten().astype(float)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

def lbp_hist(image_path, P=8, R=1, n_bins=24):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(n_bins, dtype=float)
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    hist = hist.astype(float)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

class ImageFeatureExtractor:
    def __init__(self, use_lbp=True, color_bins=(8,8,8), lbp_bins=24):
        self.use_lbp = use_lbp
        self.color_bins = color_bins
        self.lbp_bins = lbp_bins

    def transform(self, image_paths):
        feats = []
        for p in image_paths:
            ch = color_histogram(p, bins=self.color_bins)
            if self.use_lbp:
                lb = lbp_hist(p, n_bins=self.lbp_bins)
                feat = np.hstack([ch, lb])
            else:
                feat = ch
            feats.append(feat)
        feats = np.array(feats, dtype=float)
        # L2-normalize rows for stable cosine behaviour
        # If a row is all zeros (image missing) normalize will leave zeros
        if feats.shape[0] > 0:
            feats = normalize(feats, axis=1, norm='l2')
        return feats

# ---------- Similarity helper ----------
def cosine_sim_matrix(A, B):
    from sklearn.preprocessing import normalize
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    A_n = normalize(A, axis=1, norm='l2')
    B_n = normalize(B, axis=1, norm='l2')
    return np.dot(A_n, B_n.T)
