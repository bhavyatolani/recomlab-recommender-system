# 🎓 RecomLab — Recommender System Workspace

A Streamlit application for exploring user-item interactions, recommendation techniques, and recommendation analytics through an interactive dashboard.

## 📌 About the Project

RecomLab is an academic project designed to make recommender systems easier to explore. It allows users to upload interaction data, inspect dataset statistics, explore personalized recommendation examples, and view global item rankings.

The application combines data analysis with an interactive interface to provide a starting point for understanding recommendation workflows.

## ✨ Features

- 📊 **Interactive Dashboard** — View user, item, and interaction statistics.
- 📁 **CSV Upload** — Upload custom user-item rating datasets.
- 👤 **Personalized Recommendations** — Explore demo recommendations for selected users.
- 🏆 **Global Item Rankings** — Rank items by average rating and interaction count.
- 🧠 **Algorithm Selection** — Explore options for different recommendation approaches.
- 📈 **Evaluation Dashboard** — View RMSE, Precision@K, and Recall@K examples.
- 📚 **Learning Center** — Learn about datasets, algorithms, and evaluation metrics.
- 🎨 **Custom UI** — Navigate a styled dashboard with multiple tabs and sidebar controls.

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## 🧠 Recommendation Approaches

The interface provides options for:

- Collaborative Filtering
- Matrix Factorization (SVD)
- Neural Matrix Factorization (NeuMF)
- LightGCN / NGCF

**Implementation note:** These algorithms are currently represented by interface options. The application does not yet train these models.

## 📂 Project Structure

```text
RS-project/
├── app.py
├── requirements.txt
└── README.md
```

## 📊 Dataset Format

Upload a CSV containing the following required columns:

| Column | Description |
|---|---|
| `User_ID` | User identifier |
| `Item_ID` | Item identifier |
| `Rating` | Numeric rating or preference score |

Example:

```csv
User_ID,Item_ID,Rating
User_1,Item_101,5
User_1,Item_102,3
User_2,Item_101,4
User_2,Item_103,5
```

Built-in demo datasets are also available for university electives, fashion recommendations, and restaurant selection.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bhavyatolani/recomlab-recommender-system.git
```

### 2. Open the project directory

```bash
cd recomlab-recommender-system
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 📈 Evaluation Metrics

- **RMSE:** Measures the error between actual and predicted ratings.
- **Precision@K:** Measures the proportion of recommended items that are relevant.
- **Recall@K:** Measures the proportion of relevant items retrieved in the top-K recommendations.

**Current limitation:** The displayed evaluation metrics are preset illustrative values, not results calculated from actual model predictions. Personalized recommendation scores are simulated. Global item rankings are calculated from ratings in the training data.

## 🚀 Future Improvements

- Implement and train actual recommendation algorithms.
- Generate recommendations using learned user-item preferences.
- Calculate evaluation metrics from real predictions.
- Compare models using a consistent evaluation protocol.
- Add visualizations for rating distributions and user-item interactions.

## 🎯 Learning Outcomes

This project provides practical experience with Python, Streamlit, data preprocessing, user-item interaction data, recommendation concepts, and interactive analytics dashboard development.

---

**Project:** RecomLab — Recommender System Workspace  
**Status:** Academic prototype under development
