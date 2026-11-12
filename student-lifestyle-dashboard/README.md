# 🎓 Student Lifestyle & Academic Performance Dashboard

An interactive **Streamlit** dashboard that explores how daily time use (study, sleep, social life,
extracurriculars, physical activity) relates to **stress level** and **GPA** for 2,000 students.
It contains **12 visualizations**, **5 interactive filters plus per-chart controls**, and
**2 machine-learning models** (K-Means clustering and Linear Regression).

> **GitHub repository:** `<paste your public repo link here>`

---

## 1. Dataset

| Item | Detail |
|---|---|
| Source | [Student Lifestyle Dataset on Kaggle](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset) |
| Size | **2,000 rows × 8 original columns** (14 after feature engineering) |
| Numerical | `Study_Hours_Per_Day`, `Extracurricular_Hours_Per_Day`, `Sleep_Hours_Per_Day`, `Social_Hours_Per_Day`, `Physical_Activity_Hours_Per_Day`, `GPA` |
| Categorical | `Stress_Level` (Low / Moderate / High) + derived `Study_Band`, `Sleep_Band`, `GPA_Band` |
| Identifier | `Student_ID` |
| Files | `data/student_lifestyle_raw.csv` (original) · `data/student_lifestyle_clean.csv` (after preprocessing) |

**Important property of the data:** for every student the five daily-hours columns add up to *exactly* 24 h, so
`Physical_Activity_Hours_Per_Day` is effectively "whatever time is left". This is why it correlates negatively with study time,
and why it is excluded from the regression model (perfect collinearity).

## 2. Preprocessing (`preprocess.py`)

| Step | What was done | Result |
|---|---|---|
| Missing values | Numeric → median, categorical → mode (code is in place) | 0 missing values found |
| Duplicates | Checked on `Student_ID` | 0 duplicates |
| Consistency | Daily hours must total 24 h; GPA must be in 0-4 | 0 violations |
| Outliers | 1.5 × IQR rule per numeric column; rows are **flagged** (`Is_Outlier`) rather than deleted | 9 flagged rows (5 physical activity, 4 GPA); users can exclude them with a sidebar checkbox |
| Feature engineering | `Total_Hours`, `Stress_Code`, `Study_Band`, `Sleep_Band`, `GPA_Band`, `Is_Outlier` | 6 new columns |

## 3. Dashboard contents

| # | Visualization | Type | Unique analytical perspective | ML? |
|---|---|---|---|---|
| 1 | Correlation matrix | Heatmap (Pearson / Spearman toggle) | Which variables move together? | |
| 2 | GPA (or any metric) by stress level | **Violin + box plot** | Distribution shape per stress group | |
| 3 | Study/Sleep/... vs GPA density | **Density heatmap + marginal histograms** | Where students concentrate | |
| 4 | Stress → Study band → GPA band | **Treemap** | Hierarchical composition; colour = average GPA | |
| 5 | All variables per student | **Parallel coordinates** | Multivariate profile, brushing on axes | |
| 6 | Study → Sleep → Stress → GPA | **Sankey diagram** | Flow of student groups across stages | |
| 7 | Students in 3-D coloured by cluster | **3-D scatter** | Natural groups of students | ✅ K-Means |
| 8 | Group profiles | **Radar chart** | Index vs average student, per cluster / stress / study band | ✅ uses clusters |
| 9 | Actual vs predicted GPA + coefficients | Scatter + bar (sub-plots) | What drives GPA, how well can we predict it? | ✅ Linear regression |

| 10 | GPA distribution | Histogram | Frequency of GPA bands, with adjustable bins | |
| 11 | Pairwise activity comparisons | Scatterplot matrix | Compare time-use pairs within and across stress groups | |
| 12 | Stress → Sleep band → Study band | Icicle plot | Drill into the size of groups defined by stress and sleep | |

Lecture references: DataVis 4 slide 12 (histogram), DataVis 5 slide 31 (scatterplot matrix), and DataVis 6 slide 15 (icicle plot).

### Interactive components
- **Sidebar:** stress-level multiselect, study-hours slider, sleep-hours slider, GPA slider, "exclude outliers" checkbox, k-clusters slider.
- **Per-chart controls:** correlation-method radio, metric dropdown (Viz 2), x-axis dropdown (Viz 3), hierarchy-depth radio (Viz 4),
  Sankey stage multiselect (Viz 6), 3-D axis dropdowns (Viz 7), radar grouping dropdown (Viz 8), *what-if GPA predictor* sliders (Viz 9).
- **Plotly built-ins:** hover tooltips, zoom, pan, legend click-to-isolate, 3-D rotation, axis brushing, treemap drill-down.

### Machine-learning models (`ml_models.py`)
1. **K-Means clustering** on 6 standardised variables (k adjustable, 2-5; clusters are renumbered by average GPA; silhouette score is reported).
2. **Linear regression** predicting GPA from study, sleep, social and extracurricular hours (75/25 train-test split; R², RMSE and standardised coefficients shown).

## 4. Findings to discuss

- Study hours and GPA are positively associated in the sample. Compare the GPA density and the held-out regression results.
- Stress groups show different study-time profiles. Use the icicle chart to check how these groups split by sleep and study bands.
- The time-use measurements sum to 24 hours, which constrains their correlations. The charts and models describe associations in this dataset.

## 5. Setup and run

Extract the project or clone the published repository. Open PowerShell in the folder containing `app.py` and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). To regenerate the cleaned dataset, run `.\.venv\Scripts\python.exe preprocess.py`.

## 6. Project structure
```
student_dashboard/
├── app.py                # Streamlit dashboard (main file)
├── preprocess.py         # cleaning, outlier flagging, feature engineering
├── ml_models.py          # K-Means and linear regression helpers
├── requirements.txt
├── README.md
└── data/
    ├── student_lifestyle_raw.csv
    └── student_lifestyle_clean.csv
```

## 7. Team contributions

Draft allocation. Replace the placeholders and record what each person actually did before submitting.

| Member | Data preprocessing | Visualizations (design & code) | Interactivity | Other |
|---|---|---|---|---|
| **Member 1 - `<name>`** | Data quality checks, outlier detection (`preprocess.py`) | Viz 1 Correlation heatmap · Viz 2 Violin · Viz 3 Density heatmap · Viz 10 Histogram | Sidebar filters, correlation/metric/axis controls | README |
| **Member 2 - `<name>`** | Feature engineering (bands, codes) | Viz 4 Treemap · Viz 5 Parallel coordinates · Viz 6 Sankey · Viz 12 Icicle plot | Treemap depth, Sankey stage selector | One-page report |
| **Member 3 - `<name>`** | - | Viz 7 3-D scatter · Viz 8 Radar · Viz 9 Regression · Viz 11 Scatterplot matrix | k-slider, 3-D axis selectors, what-if predictor | ML models (`ml_models.py`), presentation |

## 8. Colour & design choices
One colour per entity across all charts (Low = green, Moderate = blue, High = orange); blue sequential scale for magnitude;
blue-grey-orange diverging scale for correlation. Every chart has a title, labelled axes/legends and a one-line "Insight" caption.


## 9. English presentation

1. Introduce the 2,000-student dataset. Explain the 24-hour constraint and show the outlier checkbox.
2. Compare GPA by stress group. Demonstrate one slider, legend selection on the scatterplot matrix, and drill-down on the icicle plot.
3. Show the K-Means silhouette score and explain the regression holdout. Describe one limitation and show the one-page report.

Before submitting, publish the project on GitHub, fill in the actual contributions above, and add the public URL to the report's GitHub field.
