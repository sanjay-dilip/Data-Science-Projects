# Data-Science-Projects

A collection of data analysis projects using Python and core data science techniques.

---

## 📌 Overview
This repository contains projects focused on extracting insights from structured datasets.  
Each project follows a structured workflow that includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Visualization
- Statistical insights

---

## 📊 Projects Included

### 1️⃣ MovieLens Data Analysis
- **Dataset:** [MovieLens](https://grouplens.org/datasets/movielens/)  
- **Objective:** Explore the users, ratings, and movies data to uncover meaningful insights about user behavior and movie preferences.  
- **Key Insights:**
  - 🔹 Ratings are skewed toward higher values (most ratings fall between 3–4 stars).  
  - 🔹 Popularity (most rated movies) and quality (highest average ratings) do not always overlap.  
  - 🔹 User activity follows a long-tail distribution — a small group of users rate movies very frequently.     

---

### 2️⃣ MovieLens Recommender System
- **Folder:** [movielens recommender](https://github.com/sanjay-dilip/Data-Science-Projects/tree/main/movielens-recommender)
- This project expands on the earlier MovieLens analysis and builds a full recommender system.
- **Objective:** Help users find movies that match their taste by using rating history and movie content.

The project uses a pipeline, an ALS collaborative model, a hybrid ranking model, and a Streamlit app.

**What it includes:**

- a data pipeline that prepares the full dataset

- an ALS model trained on the sparse user-item matrix

- a hybrid LightGBM ranker that uses content features and ALS scores

- an interactive Streamlit app that displays recommendations

- multiple notebooks that walk through EDA, feature building, model training, and recommendation output

This folder shows a more complete workflow that goes beyond EDA, with a pipeline, models, evaluation, and an interface.

## 📌 Future Additions
✅ More datasets with EDA explorations  
✅ Statistical modeling and hypothesis testing  
✅ Visualization dashboards and reports  
✅ Time-series and forecasting projects  
✅ end-to-end workflows similar to the MovieLens recommender
✅ small apps or dashboards to display results   

---

## ⚙️ Requirements
These projects use standard Python data science libraries.
The environment includes:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- tqdm
- lightgbm
- implicit
- streamlit

You can install missing packages using:
**pip install -r requirements.txt**
or install them one by one based on project needs.
