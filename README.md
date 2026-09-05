# Explainable Software Defect Prediction Using Static Code Metrics

This project develops an explainable machine-learning framework for predicting defective Java software modules. It compares six classification algorithms under Original and SMOTE-balanced training conditions, identifies influential static code metrics using SHAP, compares SHAP with aggregated LIME explanations, and evaluates explanation stability across repeated model runs.

The analysis uses five public PROMISE software defect datasets and is implemented as a reproducible Google Colab/Jupyter notebook.

## Research Questions

1. **RQ1 — Predictive performance:** How effectively do the machine-learning models predict software defects under Original and SMOTE conditions?
2. **RQ2 — Feature influence and consistency:** Which static code metrics are most influential, how consistent are their rankings across projects, and to what extent do SHAP and LIME agree?
3. **RQ3 — Explanation stability:** How stable are SHAP and LIME explanations across repeated runs under SMOTE conditions?

## Datasets

Five Java project datasets from the PROMISE software engineering repository are used. Each observation represents a software module or Java class.

| Dataset   | Modules | Non-defective | Defective | Defective rate |
| --------- | ------: | ------------: | --------: | -------------: |
| Ant 1.7   |     745 |           579 |       166 |         22.28% |
| Camel 1.6 |     965 |           777 |       188 |         19.48% |
| Log4j 1.2 |     205 |            16 |       189 |         92.20% |
| Poi 3.0   |     442 |           161 |       281 |         63.57% |
| Xalan 2.7 |     909 |            11 |       898 |         98.79% |

The original `bug` count is converted into a binary target:

```text
defective = 1 if bug > 0
defective = 0 otherwise
```

The projects have different imbalance directions. Defective modules are the minority in Ant and Camel, whereas non-defective modules are the minority in Log4j, Poi and Xalan.

## Static Code Metrics

The original datasets contain 20 object-oriented static code metrics describing size, complexity, coupling, cohesion and inheritance. Metadata columns and the raw defect count are excluded from modelling.

Correlation-based redundancy screening removes `dit`, `rfc`, `ic` and `max_cc`, leaving the following common 16-feature space:

```text
wmc, noc, cbo, lcom, ca, ce, npm, lcom3,
loc, dam, moa, mfa, cam, cbm, amc, avg_cc
```

## Experimental Workflow

1. Load and validate the five project datasets.
2. Inspect missing values, duplicates, invalid values and target distributions.
3. Perform exploratory analysis of skewness, correlations, class differences, statistical significance and effect sizes.
4. Convert the raw defect count into a binary target and remove metadata.
5. Apply mutual-information and correlation-based feature analysis.
6. Create a stratified 70:30 train-test split for every project.
7. Compare Original and SMOTE training conditions. SMOTE is applied only within the training pipeline.
8. Tune six classifiers using five-fold stratified cross-validation with ROC-AUC as the primary selection metric.
9. Refit the selected model and evaluate it on untouched test data.
10. Calculate full-test global SHAP importance and compare SHAP with LIME using the same fixed 50-module subset.
11. Repeat the SMOTE-XGBoost experiment across 20 random seeds to evaluate SHAP and LIME stability.

## Models

* Dummy classifier
* Logistic Regression
* Support Vector Machine
* K-Nearest Neighbours
* Random Forest
* XGBoost

Logarithmic transformation is applied to the static metrics. Logistic Regression, SVM and KNN additionally use standardisation. Hyperparameters are selected independently for each dataset and training condition using grid search.

## Evaluation Metrics

Predictive performance is assessed using:

* ROC-AUC
* F1-score
* Recall
* Precision
* Accuracy

Explanation agreement and stability are evaluated using:

* Kendall’s tau for agreement between two complete rankings
* Top-k Jaccard similarity for overlap between leading feature sets
* Kendall’s W for agreement across multiple projects or repeated runs
* Per-instance Kendall’s W for the repeatability of individual LIME explanations

## Key Results

### Model Performance

XGBoost achieved the highest overall cross-validated ROC-AUC and was selected as the champion model. Its mean ROC-AUC was **0.845 under the Original condition** and **0.836 under SMOTE**. Random Forest achieved a slightly higher mean SMOTE F1-score, but ROC-AUC was defined in advance as the primary model-selection metric.

SMOTE did not improve every dataset uniformly. It mainly improved recall and F1-score for Ant and Camel, while its effect on held-out ROC-AUC was small and dataset-dependent. This indicates that SMOTE primarily changed the classification operating point rather than consistently improving discrimination.

### Cross-Project SHAP Consistency

Cross-project SHAP agreement was weak:

* Kendall’s W: **0.294**
* p-value: **0.106**
* Mean pairwise Kendall’s tau: **0.070**
* Mean top-five Jaccard similarity: **0.183**

These findings suggest that influential static code metrics are largely project-specific. A feature identified as important for one project should not automatically be treated as equally influential for another project.

### SHAP–LIME Agreement and Stability

| Dataset   | SHAP–LIME tau | Top-5 Jaccard | SHAP stability W | LIME stability W |
| --------- | ------------: | ------------: | ---------------: | ---------------: |
| Poi 3.0   |         0.750 |         0.667 |            0.967 |            0.924 |
| Ant 1.7   |         0.400 |         0.250 |            0.862 |            0.880 |
| Camel 1.6 |         0.533 |         0.667 |            0.909 |            0.772 |
| Log4j 1.2 |         0.767 |         1.000 |            0.963 |            0.942 |
| Xalan 2.7 |         0.717 |         0.667 |            0.986 |            0.947 |

Log4j showed the strongest top-five agreement, whereas Ant showed the weakest agreement. The full-test and fixed-subset SHAP rankings remained strongly correlated across all projects (`tau = 0.817–1.000`), supporting the use of the fixed subset for the controlled SHAP–LIME comparison.

Across 20 repeated SMOTE-XGBoost runs, SHAP achieved a higher average stability than LIME:

* Mean SHAP stability: **0.937**
* Mean LIME stability: **0.893**

Xalan was the most stable project for both explanation methods. Ant was the least-stable SHAP project, while Camel was the least-stable LIME project. The leading features were generally more stable than the middle-ranked features.

The per-instance LIME analysis showed that high aggregated stability does not mean that every local explanation is equally stable. Xalan recorded a median per-instance Kendall’s W of 0.638, compared with 0.615 for Camel. Aggregated and per-instance stability are therefore reported as complementary measures.

## Interpretation Notes

* Full-test mean absolute SHAP values are used for the primary global feature-importance analysis.
* LIME is inherently local. Its reported global importance is an aggregation of absolute LIME weights across the fixed 50-module subset.
* Aggregated LIME importance should not be treated as equivalent to native global SHAP importance.
* ROC curves evaluate the underlying predictive model, not SHAP or LIME themselves.
* A high Kendall’s W may arise from deterministic model behaviour. The notebook checks whether predictions change across random seeds before interpreting stability as robustness to model variation.
* Accuracy and positive-class F1 require caution for Log4j and Xalan because defective modules are the majority class and the non-defective test samples are limited.
* Explanation stability measures repeatability rather than causality, correctness or practical usefulness.

## Repository Structure

```text
.
├── README.md
├── software_Defect_prediction_gpt-4.ipynb
└── data/
    ├── ant-1.7.csv
    ├── camel-1.6.csv
    ├── log4j-1.2.csv
    ├── poi-3.0.csv
    └── xalan-2.7.csv
```

The notebook creates the following output structure in Google Drive:

```text
Software_Defect_Prediction/
├── feature_selected_data/
└── rq1_rq3_results/
    ├── tables/
    ├── figures/
    ├── appendix/
    ├── models/
    ├── checkpoints/
    └── software_versions.csv
```

The `checkpoints` and `models` directories allow interrupted model searches and repeated-run experiments to resume without restarting completed work.

## Installation

Python 3.10 or later is recommended. Install the required packages using:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost \
            shap lime matplotlib seaborn joblib openpyxl jupyter
```

The notebook also installs the main modelling and XAI dependencies automatically when executed in Google Colab.

## Running the Project

1. Clone or download this repository.
2. Place the five PROMISE CSV files in the `data` directory.
3. Open `software_Defect_prediction_gpt-4.ipynb` in Jupyter Notebook or Google Colab.
4. If using Google Colab, upload the project directory to Google Drive and mount Drive when prompted.
5. Update `DATA_PATH` and `PROJECT_ROOT` if your project uses a different directory:

```python
DATA_PATH = "/content/drive/MyDrive/Software_Defect_Prediction/data/"
PROJECT_ROOT = Path("/content/drive/MyDrive/Software_Defect_Prediction")
```

6. Run the notebook cells in order.

Model tuning and the 20-seed stability experiment may require considerable execution time. Results are checkpointed after completed runs so that execution can resume after a disconnected session.

## Reproducibility

The main reproducibility settings are:

* Stratified 70:30 train-test splitting
* Five-fold stratified cross-validation
* Fixed train-test seed: `42`
* Fixed SHAP–LIME comparison seed: `42`
* Stability seeds: `0–19`
* Fixed explanation subset: `50` test modules per project
* LIME perturbation samples: `2,000` per explanation
* SMOTE applied only to training data
* Saved software-version information
* Saved trained models and intermediate checkpoints

## Limitations

* The study uses five Java projects from one public repository.
* Some test sets contain very few minority-class observations.
* Results depend on the selected preprocessing, models, hyperparameters and 0.5 classification threshold.
* Aggregated LIME importance depends on the selected observations and perturbation configuration.
* Explanation stability does not establish explanation fidelity or causal influence.
* The final XAI analysis evaluates only the selected XGBoost algorithm.
* The results may not generalise directly to other programming languages, systems or industrial datasets.

## Future Work

Future research could:

* Evaluate additional project releases, programming languages and industrial datasets.
* Compare SMOTE with class weighting and alternative resampling techniques.
* Assess other predictive models and explanation methods.
* Increase the number of repeated experimental runs.
* Evaluate robustness to input perturbations.
* Include explanation fidelity measures.
* Conduct developer studies of explanation usefulness and comprehensibility.

## Academic Use

This repository was developed as part of a master’s-level data science research project. If the project or datasets are reused, cite the original PROMISE data source and the relevant SHAP, LIME, SMOTE and software defect prediction literature.
