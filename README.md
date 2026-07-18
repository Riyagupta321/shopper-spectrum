# 🛍️ Shopper Spectrum — Customer Segmentation & Product Recommendation

An end-to-end data science project built during my Data Science internship at **Labmentix**. It segments e-commerce customers using **RFM analysis + KMeans clustering** and recommends products using **cosine similarity**, all wrapped in an interactive **Streamlit dashboard**.

## What it does
- Cleans and processes online retail transaction data
- Builds RFM (Recency, Frequency, Monetary) features per customer
- Segments customers into 4 groups: **High-Value, Regular, Occasional, At-Risk**
- Recommends similar products based on purchase patterns

## Tech Stack
Python · pandas · scikit-learn (KMeans, StandardScaler) · Streamlit · pickle

## Running Locally
git clone https://github.com/your-username/shopper-spectrum.git
cd shopper-spectrum
pip install -r requirements.txt
python -m streamlit run app.py

Upload the retail dataset (CSV) when prompted in the app to generate segments and recommendations.

Note: similarity_matrix.pkl (~115MB) is tracked via Git LFS. Run `git lfs install` before cloning.