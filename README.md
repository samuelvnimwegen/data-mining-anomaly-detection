# Data Mining Assignment 3 - Clustering and Anomaly Detection

This repository contains the full setup for Samuel van Nimwegen's University of Antwerp Data Mining Assignment 3.

## Project goals

- Rebuild hidden topical groups with unsupervised clustering.
- Detect unusual or unsafe files with anomaly detection.
- Keep the workflow reproducible and easy to review.
- Apply advanced text normalization with separate clustering and anomaly views.

## Repository structure

```text
data-mining-anomaly-detection/
  data/
    raw/
      articles.csv
    processed/
      normalized_text_views.csv
      clustering_tfidf_matrix.npz
      anomaly_lsa_matrix.npy
    results/
      clusters.csv
      anomalies.csv
      notebook_*.csv
  src/
    core/
      assignment_pipeline.py
      data_io.py
      paths.py
    tasks/
      preprocessing/
        text_normalization.py
        vectorizer.py
      clustering/
        kmeans_clustering.py
      anomaly_detection/
        isolation_forest_detection.py
      exploration/
        data_exploration.py
  notebooks/
    01_advanced_exploration_and_normalization.ipynb
    02_normalization_walkthrough.ipynb
  tests/
    test_pipeline.py
    test_text_normalization.py
    test_data_exploration.py
  .github/workflows/ci.yml
  main.py
  pyproject.toml
  requirements.txt
```

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

## How to run

The repository-level script is `main.py`. Inside this file you can toggle tasks by commenting or uncommenting one line.

```python
assignment_pipeline.run_clustering()
assignment_pipeline.run_anomaly_detection()
# assignment_pipeline.run_full()
```

Run it with:

```bash
python main.py
```

## Outputs

- `data/results/clusters.csv` with columns: `doc_id,label`
- `data/results/anomalies.csv` with columns: `anomaly,doc_id`
- `data/results/notebook_*.csv` files with notebook diagnostics and comparison tables
- `data/processed/normalized_text_views.csv` with cached normalized texts
- `data/processed/clustering_tfidf_matrix.npz` with cached clustering features
- `data/processed/anomaly_lsa_matrix.npy` with cached anomaly features

## Quality checks

```bash
ruff format --check .
ruff check .
pytest -q
```

The CI pipeline runs exactly these checks on pushes and pull requests.

## Advanced normalization design

- Lowercases all text before tokenization.
- Removes HTML tags, URLs, and email patterns.
- Removes stop words and corpus-specific noise terms.
- Applies lemmatization with safe fallback that keeps full words.
- Builds two views:
  - semantic view for clustering (clean vocabulary)
  - structural view for anomaly detection (keeps punctuation markers)

## Vectorization and high-dimensional representation

- `bow`: Count-based baseline for quick comparisons.
- `tfidf`: Primary clustering representation with interpretable terms.
- `tfidf_lsa_dense`: TF-IDF reduced with Truncated SVD (LSA) for dense anomaly features.
- `tfidf_svd_dense`: Backward-compatible alias that maps to `tfidf_lsa_dense`.
- Default pipeline strategy:
  - clustering uses word-level `tfidf`
  - anomaly detection uses `tfidf_lsa_dense` with char n-grams

This design keeps cluster terms interpretable while giving anomaly detection a dense feature space.

## Dimensionality reduction with LSA

- LSA applies Truncated SVD on top of TF-IDF.
- It reduces sparse high-dimensional vectors into compact dense vectors.
- It improves distance stability for anomaly detection and speeds model fitting.
- Truncated SVD works directly on sparse TF-IDF without dense centering.

## Notebooks

Use these notebooks for report material and qualitative checks:

- `notebooks/01_advanced_exploration_and_normalization.ipynb`
- `notebooks/02_normalization_walkthrough.ipynb`
- `notebooks/03_task2_clustering_methods_demo.ipynb`
- `notebooks/04_task2_cluster_interpretation_and_submission.ipynb`
- `notebooks/05_task3_anomaly_methods_and_top50.ipynb`
- `notebooks/06_task3_lsa_anomaly_search.ipynb`
- `notebooks/07_task3_anomaly_vectorization_comparison.ipynb`

## Strategic guide

- `docs/assignment_3_strategic_implementation.md` (first-person report draft for Assignment 3)
