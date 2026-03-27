# 🎬 StreamScope — Netflix Content Strategy Analyzer

A data-driven dashboard that analyzes Netflix's global content catalog to uncover trends in genre popularity, regional content, ratings distribution, content growth, and more.

---

## 📁 Project Structure

```
StreamScope/
├── data/
│   └── netflix_titles.csv        ← Place your dataset here
├── src/
│   ├── preprocessing.py          ← Data cleaning & feature engineering
│   ├── eda.py                    ← EDA helper functions
│   └── modeling.py               ← KMeans clustering + RF classification
├── app.py                        ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Download the Dataset
Download `netflix_titles.csv` from [Kaggle](https://www.kaggle.com/datasets/ranaghulamnabi/netflix-movies-and-tv-shows-dataset) and place it in the `data/` folder.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Features

| Feature | Description |
|---|---|
| **KPI Cards** | Total titles, movies, shows, countries |
| **Content Growth** | Yearly upload trend (Movies vs. TV Shows) |
| **Genre Analysis** | Top genres by count |
| **World Map** | Choropleth of content by country |
| **Rating Distribution** | Donut chart |
| **Clustering** | KMeans grouping of titles |
| **Classification** | Movie vs. TV Show prediction (RandomForest) |
| **Data Table** | Filterable raw data |

---

## 🛠 Tech Stack

- **Python 3.8+**
- `pandas`, `numpy` — data processing
- `plotly`, `seaborn` — visualization
- `scikit-learn` — ML models
- `streamlit` — interactive dashboard
