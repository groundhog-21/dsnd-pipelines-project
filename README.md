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
| Precision for recommended reviews | 0.8361 | 0.8581 |
| Precision for not-recommended reviews | 0.8824 | 0.8846 |
| Recall for not-recommended reviews | 13.3% | 27.1% |

Tuning increased correctly identified not-recommended reviews from 45 to 92 out of 339. However, 72.9% of this class remained missed. The model's high overall accuracy therefore does not imply strong detection of not-recommended reviews. Reviewers in this dataset may also differ from the retailer's wider customer base.

## Repository contents

| Path | Purpose |
|---|---|
| `starter/starter.ipynb` | Narrated analysis, model pipeline, and evaluation |
| `starter/pipeline_robustness_tests.ipynb` | Checks missing-value and unseen-category handling using a 20-review test sample |
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

## Running the notebooks

### `starter.ipynb`

**Full training:** run all cells in order. Training and tuning can take over an hour; the search uses `n_jobs=1` to avoid a parallel-processing issue encountered with spaCy.

**Recreate evaluation visuals:** run the required imports and chart-style setup, uncomment the optional CSV-loading lines and run the relevant metric and chart cells. No model fitting or model file is required. EDA charts require loading [starter/data/reviews.csv](starter/data/reviews.csv) and recreating the original train/test split.

Export commands are commented out to protect saved results. Enable them only when intentionally updating those files. Loading a local model requires the custom preprocessing functions to be defined and the spaCy language model to be loaded first.

### `pipeline_robustness_tests.ipynb`

After training in `starter.ipynb`, enable and run the model-saving cell to save `model_best` as `starter/best_model.joblib`, then comment out the saving commands again. Open `starter/pipeline_robustness_tests.ipynb` with the same environment and `starter/` working directory, and run all cells. It loads the saved pipeline without retraining and tests modified copies of a reproducible 20-review sample from the original test split. These checks assess successful handling of imperfect inputs, not prediction accuracy.

## Acknowledgments and licenses

This project builds on Udacity's starter repository, supplied dataset, and NLP pipeline lesson. ChatGPT/Codex assisted with explanations, debugging, code review, chart formatting, and editing documentation; some code snippets were adapted from that assistance. See [LICENSE.txt](LICENSE.txt) for the included Udacity educational-content terms and [fonts/OFL.txt](fonts/OFL.txt) for the Inter font license.
