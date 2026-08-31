# 🎵 Amazon Music Clustering

## 📌 Project Overview

Amazon Music Clustering is an **unsupervised machine learning project** that automatically groups songs based on their audio characteristics.

The project uses **K-Means clustering** to discover natural groups in music without using genre as an input feature.

The goal is to identify meaningful song groups that can support **playlist creation, music discovery, content organization, and recommendation systems**.

---

## 🎯 Problem Statement

Music platforms contain a large number of songs with different audio characteristics. Manually organizing these songs into meaningful categories can be difficult and time-consuming.

This project addresses the following question:

> **Can songs be automatically grouped into meaningful categories using only their audio characteristics?**

---

## 🎯 Objectives

* Explore and understand the music dataset
* Clean and validate the data
* Select relevant audio features
* Normalize the features using StandardScaler
* Apply K-Means clustering
* Determine the optimal number of clusters
* Evaluate clustering quality using the Elbow Method and Silhouette Score
* Visualize clusters using PCA
* Profile and interpret the discovered clusters
* Perform genre inference/validation after clustering
* Build an interactive Streamlit dashboard
* Identify potential business applications

---

## 📊 Dataset

**Dataset:** `single_genre_artists.csv`

| Metric              |  Value |
| ------------------- | -----: |
| Rows                | 95,837 |
| Columns             |     23 |
| Missing Values      |      0 |
| Duplicate Rows      |      0 |
| Clustering Features |     10 |
| Final Clusters      |      3 |

---

## 🎵 Features Used

The following audio characteristics were used for clustering:

* `danceability`
* `energy`
* `loudness`
* `speechiness`
* `acousticness`
* `instrumentalness`
* `liveness`
* `valence`
* `tempo`
* `duration_ms`

### Feature Purpose

| Feature          | Meaning                                |
| ---------------- | -------------------------------------- |
| Danceability     | How suitable a track is for dancing    |
| Energy           | Intensity and activity level           |
| Loudness         | Overall loudness of the track          |
| Speechiness      | Presence of spoken words               |
| Acousticness     | Likelihood of acoustic characteristics |
| Instrumentalness | Likelihood of instrumental content     |
| Liveness         | Presence of a live-performance feel    |
| Valence          | Musical positivity or mood             |
| Tempo            | Speed of the track                     |
| Duration         | Length of the track                    |

---

## 🔄 Machine Learning Workflow

```text
Data Exploration
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
K-Means Clustering
       ↓
Elbow Method
       ↓
Silhouette Score
       ↓
PCA Visualization
       ↓
Cluster Profiling
       ↓
Genre Inference / Validation
       ↓
Business Insights
       ↓
Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

The dataset was checked for quality before clustering.

### Missing Values

```text
0 missing values
```

### Duplicate Records

```text
0 duplicate rows
```

Identifiers and text-based fields were excluded from the clustering features.

Only relevant numerical audio characteristics were used.

---

## 📏 Feature Scaling

The selected features have different numerical scales.

For example:

* `tempo` is measured in BPM
* `duration_ms` is measured in milliseconds
* other audio features generally range between 0 and 1

Therefore, **StandardScaler** was applied before K-Means clustering.

This prevents features with larger numerical scales from dominating the distance calculations.

---

# 🤖 K-Means Clustering

K-Means was selected as the primary clustering algorithm.

The algorithm groups songs by minimizing the distance between observations and their assigned cluster centroid.

### Final Configuration

```text
Algorithm: K-Means
n_clusters: 3
random_state: 42
n_init: 10
```

---

# 📈 Selecting the Number of Clusters

## Elbow Method

The Elbow Method was evaluated for K values from 2 to 10.

The inertia values decreased as K increased, and the overall clustering evaluation supported selecting **K = 3**.

## Silhouette Score

|     K | Silhouette Score |
| ----: | ---------------: |
|     2 |           0.2086 |
| **3** |       **0.2465** |
|     4 |           0.2335 |
|     5 |           0.2045 |
|     6 |           0.1616 |
|     7 |           0.2049 |
|     8 |           0.1713 |
|     9 |           0.1722 |
|    10 |           0.1783 |

### Final Selection

**K = 3**

The highest silhouette score among the evaluated K values was **0.2465 at K = 3**.

---

# 📊 Cluster Results

The final K-Means model produced three clusters.

| Cluster |  Songs | Approx. % | Cluster Name                   |
| ------: | -----: | --------: | ------------------------------ |
|       0 | 12,513 |     13.1% | Spoken & Story Content         |
|       1 | 30,807 |     32.1% | Calm / Classic Music           |
|       2 | 52,517 |     54.8% | Energetic / Pop-Oriented Music |

---

# 🔍 Cluster Interpretation

## Cluster 0 — Spoken & Story Content

**Songs:** 12,513

Key characteristics:

* Speechiness: **0.830**
* Danceability: **0.664**
* Energy: **0.467**

The very high speechiness indicates a strong association with spoken, narration and story-based audio content.

---

## Cluster 1 — Calm / Classic Music

**Songs:** 30,807

Key characteristics:

* Acousticness: **0.749**
* Energy: **0.311**
* Speechiness: **0.060**

The high acousticness and lower energy indicate a calmer and more acoustic-oriented listening profile.

---

## Cluster 2 — Energetic / Pop-Oriented Music

**Songs:** 52,517

Key characteristics:

* Energy: **0.694**
* Tempo: **124.911 BPM**
* Danceability: **0.627**
* Valence: **0.666**

This is the largest cluster and represents energetic, upbeat and more dance-oriented music.

---

# 🧠 Genre Inference / Validation

Genre was **not used as an input feature during clustering**.

After creating the clusters, genre information was examined to help interpret and validate the discovered groups.

The resulting interpretations were:

```text
Cluster 0 → Hörspiel / Spoken & Story Content
Cluster 1 → Calm / Classic Music
Cluster 2 → Energetic / Pop-Oriented Music
```

This demonstrates that unsupervised clustering can discover groups that can later be interpreted using available genre information.

---

# 📉 PCA Visualization

**Principal Component Analysis (PCA)** was used to reduce the 10-dimensional feature space to two dimensions.

### Purpose

* Visualize the high-dimensional dataset
* Observe the overall cluster structure
* Understand the distribution of songs

PCA was used for **visualization only**.

The original standardized audio features were used for K-Means clustering.

---

# 🔬 Algorithm Comparison

Additional clustering algorithms were explored for comparison:

* K-Means
* DBSCAN
* Hierarchical Clustering

### DBSCAN Comparison

```text
Silhouette Score ≈ 0.7055
Davies-Bouldin Score ≈ 0.3185
Noise ≈ 0.75%
```

Although DBSCAN produced a higher silhouette score in the comparison, **K-Means was retained as the primary model** because the project specifically focuses on K-Means clustering, the Elbow Method and Silhouette Score.

---

# 💡 Business Insights

The discovered clusters can support several practical applications.

### 🎵 Automated Playlist Creation

Songs can be grouped into playlists based on their discovered audio profiles.

### 🔎 Music Discovery

Users can explore songs belonging to different audio-based groups.

### 🤖 Recommendation Systems

Cluster membership can be used as one signal for finding similar songs.

### 📂 Music Library Organization

Large music collections can be automatically organized into meaningful groups.

### 🎧 Mood-Based Listening

Different audio profiles can support energetic, calm and spoken-content listening experiences.

---

# 🖥️ Streamlit Dashboard

An interactive Streamlit dashboard was developed to explore the clustering results.

### Dashboard Pages

* 🏠 Home
* 📊 Cluster Overview
* 📈 Model Evaluation
* 🔍 Cluster Analysis
* 📈 PCA Visualization
* 🎧 Song Explorer
* 💡 Business Insights

The dashboard allows users to explore cluster sizes, audio profiles, PCA visualization and individual songs.

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib

### Dashboard

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📁 Project Structure

```text
Amazon-Music-Clustering/
│
├── Amazon_Music_Clustering.ipynb
├── app.py
├── amazon_music_cluster_profile.csv
├── amazon_music_final_clusters.csv
├── single_genre_artists.csv
├── requirements.txt
└── .gitignore
```

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/deva392000/Amazon-Music-Clustering.git
```

## 2. Navigate to the project

```bash
cd Amazon-Music-Clustering
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Run Streamlit

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

---

# 📌 Key Results

```text
Dataset Size       : 95,837 songs
Features Used      : 10
Optimal K          : 3
Silhouette Score   : 0.2465
K-Means n_init     : 10
Random State       : 42
```

### Final Cluster Distribution

```text
Cluster 0 → 12,513 songs
Cluster 1 → 30,807 songs
Cluster 2 → 52,517 songs
```

---

# 🚀 Future Improvements

Potential future enhancements include:

* Building a song recommendation engine
* Creating automatic playlist generation
* Experimenting with additional clustering algorithms
* Using dimensionality reduction techniques for improved visualization
* Incorporating additional audio characteristics
* Developing similarity-based song recommendations
* Deploying the Streamlit application online

---

# 🏁 Conclusion

This project demonstrates how **unsupervised machine learning** can be used to discover meaningful groups within a large music dataset.

Using 10 audio characteristics and K-Means clustering, **95,837 songs were grouped into three distinct audio-based clusters**.

The clusters were subsequently profiled and interpreted as:

```text
Spoken & Story Content
Calm / Classic Music
Energetic / Pop-Oriented Music
```

The project demonstrates an end-to-end machine learning workflow from **data exploration and preprocessing to clustering, evaluation, visualization, interpretation and dashboard development**.

---


