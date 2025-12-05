# src/matcher.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .features import TextFeatureExtractor, ImageFeatureExtractor, cosine_sim_matrix
import pickle
import os

class AutoMatcher:
    def __init__(self, text_weight=0.6, image_weight=0.4):
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.dataset = None
        self.text_feats = None
        self.image_feats = None

    def fit(self, df):
        """
        df: DataFrame with columns ['id','kind','description','image_path',...]
        We'll store the dataset and compute text+image features aligned to df order.
        """
        self.dataset = df.reset_index(drop=True).copy()
        all_texts = self.dataset['description'].fillna("").astype(str).tolist()
        self.text_feats = self.text_extractor.fit_transform(all_texts)
        image_paths = self.dataset['image_path'].fillna("").astype(str).tolist()
        self.image_feats = self.image_extractor.transform(image_paths)

    def match_for_item(self, item_index, top_k=5):
        """
        Given the row index (in self.dataset) for a lost (or found) item, return top_k matches
        from the opposite kind. Returns list of tuples: (candidate_index, final_score, text_sim, img_sim)
        """
        row = self.dataset.iloc[item_index]
        kind = row['kind']
        opp_mask = self.dataset['kind'] != kind
        opp_indices = np.where(opp_mask)[0]
        if len(opp_indices) == 0:
            return []

        # text similarity (tfidf sparse)
        tf_item = self.text_feats[item_index]
        tf_others = self.text_feats[opp_indices]
        # cosine_similarity handles sparse * dense correctly
        text_sim = cosine_similarity(tf_item, tf_others)[0]

        # image similarity
        img_item = self.image_feats[item_index].reshape(1, -1)
        img_others = self.image_feats[opp_indices]
        img_sim = cosine_sim_matrix(img_item, img_others)[0]

        # combine
        final_scores = self.text_weight * text_sim + self.image_weight * img_sim

        order = np.argsort(-final_scores)
        top = []
        for idx in order[:top_k]:
            cand_idx = int(opp_indices[idx])
            top.append((cand_idx, float(final_scores[idx]), float(text_sim[idx]), float(img_sim[idx])))
        return top

    def match_all(self, top_k=5):
        results = {}
        for i in range(len(self.dataset)):
            results[i] = self.match_for_item(i, top_k)
        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
