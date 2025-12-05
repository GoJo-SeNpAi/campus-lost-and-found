import streamlit as st
import pandas as pd
from src.utils import load_dataset
from src.matcher import AutoMatcher
from src.features import TextFeatureExtractor, ImageFeatureExtractor
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Campus Lost & Found AutoMatch", layout="wide")

st.title("🏫 Campus Lost & Found — AutoMatch (Classical ML)")

# Load dataset
DATA_CSV = "data/dataset.csv"
if not os.path.exists(DATA_CSV):
    st.error("Dataset not found. Run `python src/data_generation.py` to create sample data.")
    st.stop()

df = load_dataset(DATA_CSV)
st.sidebar.markdown("### Dataset")
st.sidebar.write(f"Total entries: {len(df)}")
st.sidebar.write(df[['id','kind','description']].head(10))

# Build or load matcher (build each run for simplicity)
matcher = AutoMatcher(text_weight=0.6, image_weight=0.4)
matcher.fit(df)

st.markdown("## Upload a lost or found entry (or pick from dataset)")

col1, col2 = st.columns([1,2])
with col1:
    choice = st.radio("Mode", ["Use existing entry", "Upload new"])
    if choice == "Use existing entry":
        idx = st.selectbox("Choose dataset entry", df.index.tolist(), format_func=lambda i: f"{i} | {df.loc[i,'kind']} | {df.loc[i,'description']}")
        entry_desc = df.loc[idx, 'description']
        entry_img_path = df.loc[idx, 'image_path']
        entry_kind = df.loc[idx, 'kind']
    else:
        entry_kind = st.selectbox("Entry type", ["lost","found"])
        entry_desc = st.text_input("Description", "")
        uploaded_img = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
        entry_img_path = None
        if uploaded_img:
            temp_path = os.path.join("data", "temp_upload.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_img.getbuffer())
            entry_img_path = temp_path

with col2:
    st.markdown("### Preview")
    st.write(f"**Kind:** {entry_kind}")
    st.write(f"**Description:** {entry_desc}")
    if entry_img_path and os.path.exists(entry_img_path):
        st.image(entry_img_path, use_column_width=True)
    else:
        st.write("No image provided.")

st.markdown("---")
top_k = st.slider("Top K results", 1, 10, 5)
run_button = st.button("Find matches")

if run_button:
    # If using existing dataset entry, run matcher on that index
    if choice == "Use existing entry":
        # show matches for idx
        matches = matcher.match_for_item(idx, top_k=top_k)
    else:
        # For uploaded/new entry, build temporary row and compute similarity against dataset
        tmp_df = pd.DataFrame([{
            "id": -1,
            "kind": entry_kind,
            "description": entry_desc,
            "image_path": entry_img_path if entry_img_path else ""
        }])
        
        combined = pd.concat([df, tmp_df], ignore_index=True)
        tmp_matcher = AutoMatcher(text_weight=0.6, image_weight=0.4)
        tmp_matcher.fit(combined)
        matches = tmp_matcher.match_for_item(len(combined)-1, top_k=top_k)

    if not matches:
        st.write("No matches found.")
    else:
        st.markdown("### Top matches")
        for rank, (cand_idx, score, text_sim, img_sim) in enumerate(matches, start=1):
            row = df.loc[cand_idx]
            st.write(f"**Rank {rank} — Score: {score:.3f}**  (text {text_sim:.3f}, image {img_sim:.3f})")
            cols = st.columns([1,3])
            with cols[0]:
                if os.path.exists(row['image_path']):
                    st.image(row['image_path'], width=150)
                else:
                    st.write("No image")
            with cols[1]:
                st.write(f"**ID**: {row['id']}  | **Kind**: {row['kind']}")
                st.write(f"**Description**: {row['description']}")
                st.write(f"**Object**: {row.get('object','-')}  | **Color**: {row.get('color','-')}")
                st.write("---")
