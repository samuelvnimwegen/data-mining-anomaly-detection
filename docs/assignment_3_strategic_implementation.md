# Strategic Implementation and Optimization Guide (Assignment 3)

## My Project Context

For this assignment, I work with a mixed corpus of unlabeled discussion articles (`data/raw/articles.csv`).
My goal is to:

1. Reconstruct hidden categories using clustering.
2. Detect exactly 50 corrupted or unsafe files using anomaly detection.

I built the project as a modular and reproducible pipeline, so each stage can be run independently or together.

## Repository Strategy

I organized the repository to separate responsibilities and keep experiments reproducible:

- `data/raw/`: Immutable source data.
- `data/processed/`: Cached intermediate features (normalized text views, sparse/dense matrices).
- `data/results/`: Final deliverables and notebook result tables.
- `src/`: Core pipeline code and domain modules.
- `notebooks/`: Exploratory analysis and model comparison work.
- `main.py`: Single-line toggles for running clustering and anomaly detection.

This structure lets me iterate quickly without overwriting the original data and without rerunning expensive preprocessing every time.

## Pipeline Execution Design

In `main.py`, I can run tasks by commenting or uncommenting one line:

- `assignment_pipeline.run_clustering()`
- `assignment_pipeline.run_anomaly_detection()`
- `assignment_pipeline.run_full()`

This follows the assignment requirement that task execution is easy to switch with minimal code edits.

## Task 1: Advanced Data Exploration and Normalization

I start by exploring corpus quality and noise patterns with helpers in:

- `src/exploration/data_exploration.py`
- `notebooks/01_advanced_exploration_and_normalization.ipynb`

### What I normalize and why

I apply two parallel normalization views using `TextNormalizer`:

1. **Clustering view (semantic):**
   - lowercase text,
   - remove HTML/URLs/emails,
   - remove stopwords,
   - lemmatization,
   - keep meaningful word tokens.

   I use this view to improve thematic clustering quality and term interpretability.

2. **Anomaly view (structural):**
   - preserve punctuation/symbol behavior where helpful,
   - keep patterns that can signal corruption (formatting noise, marker bursts).

   I use this view because anomalies are often structural, not only semantic.

I also export intermediate notebook tables to `data/results/` for traceable reporting.

## Vectorization and High-Dimensional Representation

I use multiple representations, with clear role separation:

- **TF-IDF (word n-grams)** for clustering interpretability.
- **TF-IDF + LSA dense embeddings** for anomaly robustness in high-dimensional noisy data.
- **Char n-grams** for anomaly modeling to capture structural artifacts.

My implementation for vectorization is centralized in:

- `src/preprocessing/vectorizer.py`

This setup gives me an interpretable cluster pipeline while still supporting strong outlier sensitivity.

## Dimensionality Reduction with LSA

I apply Truncated SVD (LSA) on TF-IDF when building dense anomaly features.

Why I use LSA:

- TF-IDF is sparse and high-dimensional.
- LSA reduces noise and stabilizes distance-based behavior.
- It improves downstream Isolation Forest behavior on noisy text.

This is documented and explored in:

- `notebooks/06_task3_lsa_anomaly_search.ipynb`
- `notebooks/07_task3_anomaly_vectorization_comparison.ipynb`

## Task 2: Clustering Methodology

My production clustering model is K-Means, implemented in:

- `src/clustering/kmeans_clustering.py`

### Model strategy

- I evaluate candidate cluster counts (bounded to assignment limits) with silhouette score.
- I keep the final number of clusters at or below 10.
- I output final labels in original row order to `data/results/clusters.csv`.

### Evaluation strategy

I combine:

1. **Quantitative evaluation:** silhouette score.
2. **Qualitative evaluation:** cluster term inspection and representative text checks via notebooks.

This combination is necessary because text clusters can look acceptable numerically while still being semantically weak.

## Task 3: Anomaly Detection Methodology

My production anomaly detector is Isolation Forest, implemented in:

- `src/anomaly_detection/isolation_forest_detection.py`

### Detection strategy

- Train Isolation Forest on anomaly-oriented features.
- Rank documents by anomaly score.
- Export anomaly IDs to `data/results/anomalies.csv`.

### About the "exactly 50" requirement

I enforce deterministic top-k selection in notebook analysis (`top 50`) for submission preparation.
This avoids fragile threshold behavior and gives consistent output size.

### Comparative diagnostics

In `notebooks/07_task3_anomaly_vectorization_comparison.ipynb`, I compare sparse TF-IDF vs TF-IDF+LSA and inspect overlap/rank-correlation behavior to understand disagreement patterns.

## Interpretability and Reporting Strategy

To maximize report quality, I document both algorithmic results and model behavior:

- Why each feature space is chosen for each task.
- How cluster count changes affect structure.
- Why anomaly overlap between feature spaces can be low.
- Which document snippets and top-ranked anomalies support my conclusions.

I avoid relying on a single metric and provide both numeric diagnostics and qualitative evidence.

## Deliverables I Produce

I keep assignment outputs versioned in the repository:

- `data/results/clusters.csv`
- `data/results/anomalies.csv`

I also keep supporting notebook exports:

- `data/results/notebook_01_*.csv`
- `data/results/notebook_02_*.csv`
- `data/results/notebook_03_*.csv`
- `data/results/notebook_04_*.csv`

This makes my final report reproducible and auditable.

## Reflection

From my perspective as a student, the key lesson is that robust text mining is not just about choosing a single model. The best results come from combining:

- clean repository architecture,
- task-specific preprocessing,
- interpretable clustering features,
- structure-aware anomaly features,
- and mixed quantitative + qualitative evaluation.

That combination gives me a defensible and high-quality Assignment 3 submission.
