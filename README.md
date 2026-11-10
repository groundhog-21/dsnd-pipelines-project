# Customer Recommendation Prediction Pipeline

An in-progress Udacity Data Science Pipeline project to predict whether a customer recommends a clothing product (`Recommended IND`: 1 = recommended, 0 = not recommended). The intended workflow combines numerical, categorical, and review-text features in a machine learning pipeline, followed by training and evaluation.

## Current status

The work is in [starter/starter.ipynb](starter/starter.ipynb). It currently:

- Loads the bundled reviews dataset and separates eight features from the target.
- Creates a shuffled, stratified 90% training / 10% test split with `random_state=27`.
- Explores target imbalance and the distribution of clothing IDs.
- Groups features for later preprocessing.

The saved notebook output reports 18,442 rows and no missing values. Its training-set output shows approximately 81.6% recommended and 18.4% not recommended. These are recorded outputs, not a fresh validation run.

**Still to implement:** preprocessing transformers, the combined pipeline, classifier selection, model fitting, evaluation, and fine-tuning. The training and fine-tuning sections are currently empty. No trained model, performance metrics, or deployment interface is included.

## Data and features

The dataset is included at [starter/data/reviews.csv](starter/data/reviews.csv); no separate download is required.

| Feature group | Columns |
| --- | --- |
| Numerical | `Age`, `Positive Feedback Count` |
| Categorical | `Clothing ID`, `Division Name`, `Department Name`, `Class Name` |
| Text | `Title`, `Review Text` |
| Target | `Recommended IND` |

The notebook describes the data as anonymized and cleaned of missing values. The repository does not document its original source or a separate dataset license.

## Setup and dependencies

Git and Python are needed for the commands below. The notebook records Python **3.11.9** in its metadata; the repository does not declare a supported Python version range.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/groundhog-21/dsnd-pipelines-project.git
cd dsnd-pipelines-project
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins:

| Package | Version |
| --- | --- |
| scikit-learn | 1.9.1 |
| NumPy | 2.4.6 |
| pandas | 3.1.1 |
| spaCy | 3.8.18 |
| Jupyter Notebook | 7.6.5 |

The current notebook imports pandas, NumPy, and scikit-learn. spaCy is listed as a dependency but is not yet used; no spaCy language model is specified. Installation and compatibility of these pins have not been verified for this guide.

## Run and check the notebook

From the repository root, with the environment active:

```bash
cd starter
python -m notebook starter.ipynb
```

Select the virtual environment's Python kernel, then restart the kernel and run all cells. Running from `starter/` matters because the notebook reads `data/reviews.csv` using a relative path.

**Testing:** no automated test suite, test configuration, or CI workflow is committed, so there is no repository-defined automated testing command. Running all notebook cells is the current manual execution check; it does not train or evaluate a model yet.

Some saved exploration outputs differ from the current cell source or execution counts. Re-run the notebook before relying on those outputs.

## Project structure

| Path | Purpose |
| --- | --- |
| `README.md` | Project guide |
| `requirements.txt` | Pinned Python dependencies |
| `starter/starter.ipynb` | Project instructions and work-in-progress analysis |
| `starter/data/reviews.csv` | Bundled reviews dataset |
| `starter/README.md` | Original scaffold-folder note |
| `.gitignore` | Excludes the virtual environment, notebook checkpoints, and local files |
| `CODEOWNERS` | Repository ownership configuration |
| `LICENSE.txt` | Udacity educational-content license terms |

## License

See [LICENSE.txt](LICENSE.txt) for the included Udacity educational-content terms.
