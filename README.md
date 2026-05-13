# ExtraaLearn — Lead Conversion Prediction

A machine learning project that builds a classification model to identify which **EdTech leads** are most likely to convert into paid customers. Built as a capstone project for my **Data Science certification**, the project uses Decision Trees and Random Forests — both in baseline and hyperparameter-tuned versions — to predict lead conversion and surface the factors that drive it.

**Author:** Carolina Ochoa 

---

## Project Overview

The EdTech industry has grown enormously, and companies like the fictional **ExtraaLearn** generate massive volumes of leads every day. The business problem is simple but high-impact: with limited sales and marketing resources, which leads should the team focus on?

This project tackles that question by:

1. Performing thorough **EDA** on a real-world-style lead dataset.
2. Engineering and preparing the data for classification modeling.
3. Building a **Decision Tree** and a **Random Forest**, each with a **baseline** and a **hyperparameter-tuned** version.
4. Evaluating models with **recall as the priority metric**, because missing a real lead (false negative) costs more than wasting effort on a non-converter (false positive).
5. Extracting **feature importances** to drive actionable business recommendations.

**Final outcome:** The **tuned Random Forest** was selected as the best model — it preserved a strong recall (0.85) while keeping false positives lower than the tuned Decision Tree, offering the best overall balance for the business.

---

## Business Objective

As a data scientist at ExtraaLearn, the goal is to:

- Build an ML model that identifies leads most likely to convert.
- Find the **factors** driving the lead conversion process.
- Build a **profile** of high-converting leads to guide marketing and sales decisions.

---

## Dataset

- **File:** `ExtraaLearn.csv`
- **Type:** Tabular lead-interaction data.
- **Target variable:** `status` — binary flag indicating whether the lead converted to a paid customer.
- **Class balance:** Imbalanced (most leads do not convert), which is why recall on the positive class is the priority metric.

**Feature dictionary:**

| Feature | Description |
|---|---|
| `ID` | Unique lead identifier (dropped during preprocessing) |
| `age` | Age of the lead |
| `current_occupation` | Professional / Unemployed / Student |
| `first_interaction` | Website / Mobile App |
| `profile_completed` | Low (0–50%) / Medium (50–75%) / High (75–100%) |
| `website_visits` | Number of website visits |
| `time_spent_on_website` | Total time spent on the site |
| `page_views_per_visit` | Average pages viewed per visit |
| `last_activity` | Email / Phone / Website activity |
| `print_media_type1` | Saw newspaper ad (flag) |
| `print_media_type2` | Saw magazine ad (flag) |
| `digital_media` | Saw digital ad (flag) |
| `educational_channels` | Heard via educational channels (flag) |
| `referral` | Came via referral (flag) |
| `status` | **Target** — converted or not |

---

## Tech Stack

**Language:** Python 3
**Environment:** Google Colab

**Core libraries:**
- `pandas` and `numpy` — data manipulation and numerical work
- `matplotlib` and `seaborn` — visualization (histograms, boxplots, heatmaps, stacked bar charts)
- `scikit-learn` — modeling (`DecisionTreeClassifier`, `RandomForestClassifier`), preprocessing (`MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`, `SimpleImputer`), evaluation (`classification_report`, `confusion_matrix`, `recall_score`, `precision_score`, `f1_score`, `roc_auc_score`), and tuning (`GridSearchCV`, `StratifiedShuffleSplit`)

---

## Data Cleaning and Preprocessing

### Step 1 — Sanity checks
- Verified shape, dtypes, and ran `.head()` / `.tail()` / `.describe()`.
- Confirmed **no duplicates** and **no missing values** — the dataset was clean out of the box.

### Step 2 — Drop the ID column
The `ID` column carries no predictive signal — it's a unique identifier per lead. Dropping it prevents the model from learning meaningless patterns and reduces dimensionality after encoding.

### Step 3 — Work on a copy
Created `datatest = data.copy()` to avoid accidentally modifying the original DataFrame. Small but important habit.

### Step 4 — Outlier strategy
Boxplots revealed clear outliers in `website_visits` and `page_views_per_visit`. **These were intentionally kept** because:
- The chosen models (Decision Trees and Random Forests) are **tree-based** and robust to outliers — they split on thresholds, not magnitudes.
- These outliers likely represent highly engaged users, which may carry real predictive signal.

### Step 5 — Categorical encoding
Used `pd.get_dummies(X, drop_first=True)` for one-hot encoding of categorical features. `drop_first=True` avoids the dummy-variable trap and keeps the feature matrix lean.

### Step 6 — Train/test split with stratification
Used `StratifiedShuffleSplit` (test_size=0.3, random_state=1) instead of plain `train_test_split` to **preserve the class distribution** in both training and test sets. With an imbalanced target, this prevents the test set from accidentally over- or under-sampling the minority class.

---

## Exploratory Data Analysis (EDA)

### Univariate findings
- **Age** — Left-skewed; majority of leads are over 50. No minors (min age 18, max 63).
- **Website visits** — Right-skewed with outliers above 10. Mean and mode both under 5. 174 leads have **never** visited the website.
- **Time spent on website** — Highly right-skewed. Average around 724 seconds (~12 minutes).
- **Page views per visit** — Many unique values; ~52% of values are unique, suggesting limited predictive power on its own.
- **Profile completion** — Most leads complete High or Medium; Low is rare.
- **First interaction** — Website is the dominant channel, followed by Mobile App.

### Bivariate findings (against `status`)
- **Time spent on website** is the strongest individual predictor of conversion.
- **Profile completion = High** correlates strongly with conversion — completing a profile signals serious intent.
- **First interaction = Website** converts noticeably better than Mobile App, which informs where to invest UX effort.
- **Referrals**, although a small share of leads (~2.3%), have the **highest conversion rate** by far — friends-and-family trust converts.
- **Print media (newspaper, magazine), digital media, and educational channels** all hover around the same ~0.3 conversion rate when looked at in isolation — they bring in leads, but don't strongly differentiate converters.

### Multivariate
- Correlation heatmap showed a meaningful relationship between `time_spent_on_website` × `age` × `status`.
- Boxplots of age by occupation behaved as expected: students cluster below 25, while professionals and unemployed leads span the older range with similar medians.

---

## Models Implemented

For each model, I implemented **both a baseline and a hyperparameter-tuned version** to demonstrate the value of tuning.

### Model 1 — Decision Tree Classifier

**Baseline:** `DecisionTreeClassifier(random_state=1, max_depth=8)`
- **Training:** recall 0.87, F1 0.83, accuracy 0.90.
- **Test:** recall dropped to 0.79, precision (class 1) dropped to 0.72 — clear sign of mild **overfitting**.

**Tuned with GridSearchCV:**
- `max_depth`: 2–9
- `criterion`: gini, entropy
- `min_samples_leaf`: 5, 10, 20, 25
- `class_weight = {0: 0.3, 1: 0.7}` — manually upweighted the positive class so the model is penalized more for missing real leads
- Scorer: `make_scorer(recall_score, pos_label=1)` — directly optimized for what the business cares about
- **Result:** training recall 0.91, test recall 0.89 — significant improvement and much less gap between train and test, meaning less overfitting. False-positive rate did rise slightly.

### Model 2 — Random Forest Classifier

**Baseline:** `RandomForestClassifier(random_state=1)`
- **Training:** near-perfect scores — a classic "too good to be true" signal of **severe overfitting**.
- **Test:** recall dropped to 0.73 — not acceptable when recall is the priority.

**Tuned with GridSearchCV:**
- `n_estimators`: 110, 120
- `max_depth`: 6, 7
- `min_samples_leaf`: 20, 25
- `max_features`: 0.8, 0.9
- `max_samples`: 0.9, 1.0
- `class_weight`: "balanced" or `{0: 0.3, 1: 0.7}`
- `criterion = "entropy"`
- **Result:** test recall jumped to **0.85**, with noticeably fewer false positives than the tuned Decision Tree.

### Model Selection

| Model | Recall (class 1) | False-Positive Rate | Overfitting |
|---|---|---|---|
| Decision Tree (baseline) | 0.79 (test) | Moderate | Mild |
| Decision Tree (tuned) | **0.89** | Higher | Reduced |
| Random Forest (baseline) | 0.73 (test) | Low | Severe |
| **Random Forest (tuned) — Best** | **0.85** | **Lowest of the tuned models** | Controlled |

**Winner: Tuned Random Forest.** The Decision Tree achieved slightly higher recall, but the Random Forest had fewer false positives, better generalization, and a wider set of feature importances — meaning it captured a richer set of signals from the data.

---

## Why Recall, Not Accuracy?

For ExtraaLearn, a **false negative** (predicting a lead won't convert when they actually would) is the worst outcome — the company misses revenue and the sales team never reaches out. A **false positive** (predicting conversion that doesn't happen) costs only some outreach time.

That asymmetry is why:
- **Recall on class 1** was the primary scoring metric in `GridSearchCV`.
- **`class_weight`** was tuned to upweight the positive class.
- **Accuracy was deliberately not the headline metric**, because on imbalanced data a model that predicts "won't convert" for everyone would look ~80% accurate while being useless.

---

## Key Feature Importances

Both tuned models agreed on the top predictors of conversion:

1. **Time spent on website** — by far the most important.
2. **First interaction (website)** — leads whose first touchpoint was the website convert better.
3. **Profile completed (high)** — serious intent signal.
4. **Last activity (website / phone)** — recent engagement matters.
5. **Age** — older professional leads are more likely to convert.

The Random Forest additionally picked up signals from `educational_channels` and `current_occupation`, which is part of why its predictions are more balanced than the single Decision Tree.

---

## Actionable Business Recommendations

The model isn't just a number — it points to concrete things ExtraaLearn can do:

- **Invest in the website experience.** Time on site and first-interaction-via-website are the two strongest predictors. UX improvements that increase session length directly map to conversions.
- **Build a referral program.** Referrals are only ~2.3% of leads but have the highest conversion rate. Incentivizing referrals is a high-ROI move.
- **Cut magazine spend.** Magazine ads convert at the same low rate as everything else and contribute the fewest leads. Reallocate that budget.
- **Double down on educational channels.** Educational channels were the second-best converter and are scalable through partnerships with schools and online learning communities.
- **Encourage profile completion.** "High" profile completion strongly predicts conversion — prompt users to finish their profiles with progress bars, incentives, or gated content.

---

## Project Structure

```
.
├── README.md                              <- You are here
├── FullCodeCarolinaOchoa.ipynb            <- Main Jupyter notebook
├── FullCodeCarolinaOchoa.html             <- Exported HTML view of the notebook
└── data/
    └── ExtraaLearn.csv                    <- Lead dataset (place here)
```

---

## How to Run

1. **Clone the repository.**
```bash
   git clone https://github.com/<your-username>/extraalearn-lead-conversion.git
   cd extraalearn-lead-conversion
```

2. **Place the dataset.** Put `ExtraaLearn.csv` in the `data/` folder.

3. **Open the notebook** in **Google Colab** (recommended) or local Jupyter.

4. **Install dependencies** (if running locally):
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
```

5. **Run all cells.** GridSearchCV steps take a few minutes — that's normal.

---

## Practices and Concepts Demonstrated

This project showcases the following data science practices, all standard in production ML work:

- End-to-end ML pipeline: ingestion → cleaning → EDA → preprocessing → modeling → evaluation → business recommendations.
- **Stratified train/test split** to preserve class balance on imbalanced data.
- **Metric-aware modeling** — choosing recall (not accuracy) because the business cost of false negatives is higher.
- **Class weighting** to handle imbalance at the algorithm level.
- **Hyperparameter tuning** with `GridSearchCV` and a **custom recall scorer** scoped to the positive class.
- Baseline-first methodology — every tuned model is compared to its untuned counterpart.
- **Overfitting diagnosis** by comparing train vs. test metrics (caught both the Decision Tree's mild overfit and the Random Forest's severe overfit).
- **Feature importance analysis** translated into business recommendations.
- Reproducibility — fixed `random_state` throughout, documented dependencies, exported HTML notebook.
- Reusable helper functions (`hist_box`, `barp`, `metrics_score`) to keep the notebook DRY.

---

## Conclusion and Future Work

The **tuned Random Forest** model successfully identifies high-converting leads with strong recall (0.85), giving ExtraaLearn a useful tool to prioritize sales effort. Beyond raw prediction, the project surfaces clear marketing recommendations: invest in the website, build a referral program, kill magazine spend, lean into educational channels, and encourage profile completion.

**Things I would improve next:**
- Try **Gradient Boosting** models (XGBoost, LightGBM, CatBoost) — typically the strongest performers on tabular data like this.
- Use **SMOTE or class-weighted resampling** for an alternative angle on the imbalance problem.
- Add a **probability-threshold analysis** — instead of the default 0.5 cutoff, tune the threshold on the precision–recall curve to optimize for the business cost ratio explicitly.
- Build a **lead-scoring dashboard** (Streamlit) that lets the sales team enter lead attributes and get a conversion probability.
- Add **SHAP values** for per-lead explanations — useful for sales conversations.

---

## Reflections

This was my first end-to-end classification project where the metric choice mattered more than the model choice. I started thinking accuracy was the goal and quickly realized that on imbalanced lead data, accuracy is misleading — a model could "achieve" 80% accuracy by predicting nobody converts. Switching the lens to recall (and then balancing it against false positives) changed how I evaluated every model. The biggest surprise was how much the **baseline Random Forest overfit** — it looked perfect on training and mediocre on test. Catching that and fixing it with tuning was the most useful lesson of the project.

---

## License

This project is released for educational and portfolio purposes.
