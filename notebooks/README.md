# Notebooks

This directory is reserved for Jupyter/IPython notebooks used during exploratory data analysis (EDA), visualization prototyping, and ad-hoc investigation.

## Suggested Use

- **EDA Notebooks**: Explore distributions, outliers, and data quality before committing to pipeline logic.
- **Visualization Drafts**: Prototype Matplotlib/Seaborn charts before moving final versions into scripts.
- **Model Comparisons**: Compare VADER vs. DistilBERT sentiment outputs, or test alternative theme extraction methods.

## Guidelines

- Notebooks should **not** contain production pipeline logic — that belongs in `src/` or `scripts/`.
- Clear all cell outputs before committing to keep diffs clean.
- Name notebooks descriptively: `01_eda_review_distributions.ipynb`, `02_sentiment_comparison.ipynb`.
