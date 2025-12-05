# Campus Lost & Found — AutoMatch (Classical ML)

This project demonstrates a feature-based lost & found matching system using TF-IDF (text) and classic image features (color hist + LBP). No deep learning.

## Run
1. Install requirements: `pip install -r requirements.txt`
2. Create demo data: `python src/data_generation.py`
3. Run Streamlit: `streamlit run app.py`

## Structure
- `src/` : code for data generation, features, matcher, evaluation
- `data/` : contains generated dataset and images

