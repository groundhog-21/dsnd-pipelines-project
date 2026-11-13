# Fashion Forward: Predicting Product Recommendations

StyleSense's rapid growth has brought a backlog of reviews with missing recommendation labels. This Udacity Data Science Pipelines project uses labeled clothing reviews to predict whether a customer recommends a product from review text, customer age, and product information.

The [notebook](starter/starter.ipynb) contains the analysis, pipeline implementation, tuning, evaluation, and visualizations.

## Data and approach

The supplied dataset contains 18,442 reviews, eight input features, and the binary target `Recommended IND` (1 = recommended; 0 = not recommended). A stratified 90/10 split preserves class proportions, with `random_state=27` for reproducibility. Approximately 81.6% of training reviews recommend the product.

The scikit-learn pipeline combines:

- Numerical imputation and standard scaling.
- Categorical imputation and one-hot encoding.
- Concatenated Title and Review Text, processed with spaCy lemmatization and TF-IDF.
- Character count of the original Review Text.
- A Random Forest classifier.

RandomizedSearchCV evaluated four parameter combinations using three-fold stratified cross-validation and macro F1. Preprocessing caching reduces repeated computation. The selected configuration uses 150 trees and 250 candidate features per split; its mean validation macro F1 was 0.6693.

## Results

Both models were evaluated on the same 1,845 held-out test reviews.

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 0.8374 | 0.8596 |
| Macro F1 | 0.5699 | 0.6678 |
| Recall for not-recommended reviews | 13.3% | 27.1% |

Tuning increased correctly identified not-recommended reviews from 45 to 92 out of 339. However, 72.9% of this class remained missed. The model's high overall accuracy therefore does not imply strong detection of dissatisfied reviewers. Reviewers in this dataset may also differ from the retailer's wider customer base.

## Repository contents

| Path | Purpose |
|---|---|
| `starter/starter.ipynb` | Narrated analysis, model pipeline, and evaluation |
| `starter/data/reviews.csv` | Udacity-supplied review data |
| `starter/cv_results.csv` | Saved cross-validation results |
| `starter/model_predictions_comparison.csv` | Test labels, baseline/tuned predictions, and original row indices |
| `starter/my_charts.mplstyle` | Shared Matplotlib chart styling |
| `fonts/` | Inter font files and their license |
| `requirements.txt` | Pinned dependencies, including the spaCy English model |

The fitted `starter/best_model.joblib` and `starter/pipeline_cache/` are local artifacts excluded from Git. They are not provided by cloning this repository.

## Setup

Developed on Windows 11 using Python 3.11.9 and VS Code with Python and Jupyter support. In PowerShell:

```powershell
git clone https://github.com/groundhog-21/dsnd-pipelines-project.git
cd dsnd-pipelines-project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `starter/starter.ipynb` in VS Code and select the `.venv` kernel. Use `starter/` as the notebook working directory: data, saved results, and style paths are relative to it. Run the font-registration and style-loading cell before plotting; it registers the bundled Inter fonts without a system-wide installation.

## Running the notebook

**Full training:** run the setup, preprocessing, training, and search cells in order. In the best-estimator cell, enable `model_best = param_search.best_estimator_` and disable the alternative `joblib.load(...)` line. The current notebook defaults to loading the locally saved model. Training and tuning can take tens of minutes; the search uses `n_jobs=1` to avoid a parallel-processing issue encountered with spaCy.

**Recreate evaluation visuals:** load the two saved results CSVs and run the relevant metric and chart cells. No model fitting or model file is required. EDA charts require loading [starter/data/reviews.csv](starter/data/reviews.csv) and recreating the original train/test split.

Export commands are commented out to protect saved results. Enable them only when intentionally updating those files. Loading a local model requires the custom preprocessing function definitions to be available first.

## Acknowledgments and licenses

This project builds on Udacity's starter repository, supplied dataset, and NLP pipeline lesson. ChatGPT/Codex assisted with explanations, debugging, code review, chart formatting, and editing documentation; some code snippets were adapted from that assistance. See [LICENSE.txt](LICENSE.txt) for the included Udacity educational-content terms and [fonts/OFL.txt](fonts/OFL.txt) for the Inter font license.
