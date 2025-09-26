💄✨ Hybrid Sentiment Analysis — Fashion & Cosmetics 

app: https://hybrid-sentiment-three-gkuwzftv6yqnru825cdxhp.streamlit.app/ 

Analyze brand sentiment for fashion & cosmetics reviews. The app blends lexicon signals + review metadata into a simple, explainable “hybrid” label, then lets you slice, explore keywords, track trends, and export product-level rollups — all in a beautiful Streamlit UI.
Live mode: The public app now runs in Fast mode only (a small curated sample for reliability). Users can also upload their own CSV in the sidebar to analyze it instantly.

🚀 What this app does
Interactive filters for brand, price range, and verified purchases
Label distribution table + chart (positive / neutral / negative or your labels)
Top keywords by sentiment (with optional wordclouds)
Brand leaderboard (positive share, avg rating, review count)
Price bands vs positive share (quantile bands)
Rating histogram
Product-level rollups (avg rating, #reviews, positive share)
→ Download CSV
Top-N products chart + CSV export
Trends over time (monthly, if a timestamp column is present)
Single review prediction (demo) for quick sanity checks
Download filtered dataset with the hybrid labels

📦 Data used (and where it came from)

This project was built using public e-commerce review datasets for fashion & cosmetics products:
Amazon Fashion 800K User Reviews Dataset (Kaggle)
A large-scale dataset of user reviews for fashion products on Amazon (~800k reviews).
Cosmetics and Beauty Products Reviews — Top Brands (Kaggle)
Reviews of beauty & cosmetics products across major global brands.
Merged and cleaned these into your data/processed/ folder:

combined_raw.csv → raw merged reviews
combined_clean.csv → cleaned/normalized fields
combined_lexicon.csv → lexicon features / sentiment hints
combined_hybrid.csv → final processed table used for analysis

For the public app (to stay within platform limits), we ship only:
data/processed/combined_demo.csv → a small, representative sample created from combined_hybrid.csv using reservoir sampling (see below).

🧪 How the demo sample was created
We used a tiny, reliable sample to keep the cloud app snappy:
data/processed/combined_hybrid.csv
        └─ reservoir sampling (e.g., 30,000 rows)
            └─ data/processed/combined_demo.csv
Script (already in your repo at scripts/create_demo_sample.py) uses reservoir sampling so the sample is unbiased across the file:

# From repo root
python scripts/create_demo_sample.py \
  --source data/processed/combined_hybrid.csv \
  --out    data/processed/combined_demo.csv \
  --rows   30000


You can change --rows to adjust demo size.
Only combined_demo.csv is tracked in Git; large CSVs are intentionally not pushed to avoid LFS quota issues.

📁 Expected columns (auto-detected)
The app auto-detects common names for each field:
Purpose	Examples it looks for
Text	review_text, text, content, body, title
Label	hybrid_label, review_label, target, label
Brand	brand_name, brand
Product ID	product_id, asin, parent_asin
Product name	product_title, title
Rating	rating, review_rating, product_rating
Price/MRP	price, mrp
Verified	verified_purchases, is_a_buyer, verified_purchase
Timestamp	timestamp, review_date, date

🧠 What “hybrid” means here
This app’s label column (e.g., hybrid_label) comes from simple, explainable rules combining:
Lexicon cues (positive/negative terms, intensifiers, negations)
Lightweight metadata (ratings, verified flag, etc.)
It’s intentionally transparent and fast to compute — good for exploratory dashboards and product/brand rollups.

🖥 Run locally
Clone the repo and create a virtual environment
pip install -r requirements.txt
Put data in data/processed/
For the included demo: combined_demo.csv (already present)
To analyze your own CSV, just upload it in the sidebar.
Launch
streamlit run app.py
Open http://localhost:8501
.

☁️ Deploying (notes)
The public Streamlit Cloud app runs Fast mode only (using combined_demo.csv or an uploaded CSV).
Large full datasets are not pushed to Git to avoid LFS quotas.

🔍 How to use the app
Start with the demo (Fast mode) and explore the dashboard.
Use the filters (brand, price range, verified) to focus your analysis.
Open Top keywords to see the dominant terms by sentiment (enable wordclouds if you like).
Check the Brand leaderboard and Price bands to see what correlates with positivity.
Scroll to Product-level rollups: sort by positive share or rating, export CSV, and view Top-N products (chart + CSV).
If your data has dates, explore Trends over time.
Use Download filtered dataset to take your cut downstream.
To analyze your own file: upload CSV in the sidebar — it instantly overrides the demo.

🧰 Requirements

Key packages (see requirements.txt):
streamlit, pandas, numpy, scikit-learn, nltk, matplotlib, altair, tqdm, requests
wordcloud (optional; enables wordclouds in the keywords section)
Install:
pip install -r requirements.txt

📚 Repo layout
.
├─ app.py                               # Streamlit app (Fast mode only)
├─ requirements.txt
├─ scripts/
│  └─ create_demo_sample.py             # reservoir sample -> combined_demo.csv
├─ data/
│  └─ processed/
│     └─ combined_demo.csv              # small demo sample (tracked in Git)
└─ src/                                 # (your earlier pipeline code)
   ├─ preprocess.py / ingest_merge.py / lexicon_sentiment.py / ...
   └─ utils.py

## 📸 Screenshots

<!-- Main Highlight -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/e2952226-2c0b-444f-8fbb-cd6daa8313ac" 
       alt="Homepage dashboard view" width="850"/>
</p>

<!-- Side by side 1 -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f029365-2585-498f-9942-3d642373d144" 
       alt="Sentiment analysis results" width="420"/>
  <img src="https://github.com/user-attachments/assets/ec1e71a2-941d-421e-a6d3-5b8e0d4b66f9" 
       alt="Data preprocessing workflow" width="420"/>
</p>

<!-- Side by side 2 -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/bf178ba0-4681-4bec-a373-d114fe09729c" 
       alt="User review input and prediction" width="420"/>
  <img src="https://github.com/user-attachments/assets/213c4889-f594-4462-aba0-ffb7158567c0" 
       alt="Feature engineering and cleaning step" width="420"/>
</p>

<!-- Side by side 3 -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/4108e9d0-4c64-4927-a45a-c885da93cab9" 
       alt="Model performance metrics" width="420"/>
  <img src="https://github.com/user-attachments/assets/77ad05b0-42c6-49c0-a05e-8c0a0dcf955b" 
       alt="Detailed analysis report view" width="420"/>
</p>

<!-- Side by side 4 -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/f348075c-b257-4700-bcc7-d0aa90952953" 
       alt="Final results and insights dashboard" width="420"/>
</p>


🔒 Privacy & limitations


This app is exploratory. The “hybrid” label is simple and explainable — great for dashboards and rollups, not a replacement for fully trained models.
Public demo uses a small sample, so counts/ratios are indicative. For production decisions, run locally on your full data and validate.

📝 Citation / credit

App: Hybrid Sentiment Analysis — Fashion & Cosmetics (Streamlit)
Author: Diksha Jain
Data Sources:
Amazon Fashion 800K User Reviews Dataset (Kaggle)
Cosmetics and Beauty Products Reviews — Top Brands (Kaggle)
