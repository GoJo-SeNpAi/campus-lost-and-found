# src/evaluate.py
import pandas as pd
import numpy as np
from .matcher import AutoMatcher

def create_ground_truth(df):
    """
    For synthetic data, treat rows with same object + color pair (lost vs found) as a true match.
    Returns dict: lost_index -> set(found_indices)
    """
    gt = {}
    for i, row in df.iterrows():
        if row['kind'] == 'lost':
            matches = df[(df['kind']=='found') & (df['object']==row['object']) & (df['color']==row['color'])].index.tolist()
            gt[i] = set(matches)
    return gt

def topk_accuracy(matcher, df, k=1):
    gt = create_ground_truth(df)
    correct = 0
    total = 0
    for lost_idx in gt.keys():
        total += 1
        preds = matcher.match_for_item(lost_idx, top_k=k)
        pred_indices = [p[0] for p in preds]
        if any([p in gt[lost_idx] for p in pred_indices]):
            correct += 1
    return correct / total if total>0 else 0

def mrr(matcher, df, max_k=10):
    gt = create_ground_truth(df)
    rr_sum = 0
    total = 0
    for lost_idx, relevant in gt.items():
        total += 1
        preds = matcher.match_for_item(lost_idx, top_k=max_k)
        found_rank = None
        for rank, (cand_idx, *_ ) in enumerate(preds, start=1):
            if cand_idx in relevant:
                found_rank = rank
                break
        if found_rank:
            rr_sum += 1.0 / found_rank
    return rr_sum / total if total>0 else 0
