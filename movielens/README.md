# MovieLens Movie Ratings – EDA & Recommender (Python)

## 📌 Overview
This project explores the **MovieLens** dataset to understand user–movie interaction patterns and build a simple **recommendation baseline**.  
It covers data loading, cleaning, exploratory data analysis (EDA), and a first-pass recommender using collaborative signals.

---

## 🗂️ Data
- **Source:** MovieLens (GroupLens) – commonly used `ratings.csv`, `movies.csv` (and optionally `links.csv`, `tags.csv`).
- **Typical columns:**
  - `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`
  - `movies.csv`: `movieId`, `title`, `genres`
- **Notes:** If you’re running locally, download a version (e.g., **ml-latest-small**) from GroupLens and place the CSVs in a `data/` folder.

---

## 🔎 What’s Inside the Notebook
1. **Data Loading & Cleaning**
   - Read ratings & movies, handle missing values, ensure correct dtypes, basic sanity checks.
2. **Exploratory Data Analysis**
   - Ratings distribution, per-user/per-movie counts, temporal trends.
   - Top-rated and most-rated movies; popularity vs. average rating.
3. **Baseline Recommenders**
   - **Global mean / movie mean** baseline (simple, transparent).
   - (Optional) **User-mean / Z-score** normalization preview.
4. **Evaluation (Lightweight)**
   - Holdout split for quick validation (e.g., train/test by timestamp).
   - Simple metrics (MAE/RMSE) for baselines.
5. **(Optional) Next Steps**
   - Matrix factorization (e.g., Surprise SVD) or Implicit ALS.
   - Content-based features from genres; hybrid scoring.
