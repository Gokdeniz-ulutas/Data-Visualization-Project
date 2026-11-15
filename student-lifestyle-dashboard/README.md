# 🎓 Student Lifestyle & Academic Performance Dashboard

An interactive **Streamlit** dashboard that explores how daily time use (study, sleep, social life,
extracurriculars, physical activity) relates to **stress level** and **GPA** for 2,000 students.
It contains **12 visualizations**, **5 interactive filters plus per-chart controls**, and
**2 machine-learning models** (K-Means clustering and Linear Regression).

> **GitHub repository:** `https://github.com/Gokdeniz-ulutas/Data-Visualization-Project`

---

## 1. Dataset

| Item | Detail |
|---|---|
| Source | [Student Lifestyle Dataset on Kaggle](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset) |
| Size | 2,000 rows × 8 original columns (14 after feature engineering) |
| Numerical | `Study_Hours_Per_Day`, `Extracurricular_Hours_Per_Day`, `Sleep_Hours_Per_Day`, `Social_Hours_Per_Day`, `Physical_Activity_Hours_Per_Day`, `GPA` |
| Categorical | `Stress_Level` (Low / Moderate / High) + derived `Study_Band`, `Sleep_Band`, `GPA_Band` |
| Identifier | `Student_ID` |
| Files | `data/student_lifestyle_raw.csv` (original) · `data/student_lifestyle_clean.csv` (after preprocessing) |

Important property of the data: for every student the five daily-hours columns add up to *exactly* 24 h, so
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

| # | Visualization | Type | Unique analytical perspective | ML |
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

| Member | Charts designed and implemented | Additional contributions |
|---|---|---|
| **Member 1 - `<Yunus-2022556046>`** | **1–3** — Correlation heatmap, violin with box plot, and 2D density heatmap | Checked missing values, duplicate IDs, GPA ranges and daily totals, and flagged outliers. Set up the correlation method, metric and axis selectors. |
| **Member 2 - `<Selim-2022556065>`** | **4–6** — Treemap, parallel coordinates, and Sankey diagram | Prepared the study, sleep and GPA bands. Set up treemap drill-down, axis brushing and the Sankey stage selector. |
| **Member 3 - `<Gökdeniz-2022556063>`** | **7–9** — 3D scatter, radar chart, and regression scatter with coefficient bars | Handled K-Means and linear regression. Added the cluster-count slider, 3D axis selectors, radar grouping and what-if GPA controls. |
| **Member 4 - `<Serdar-2022556052>`** | **10–12** — GPA histogram, scatterplot matrix, and icicle plot | Adapted three charts from the course slides. Set up the histogram bin slider, stress-group legend controls and icicle drill-down. Wrote the one-page report. |

Each member presents their three charts and demonstrates one interaction.

## 8. Colour & design choices
One colour per entity across all charts (Low = green, Moderate = blue, High = orange); blue sequential scale for magnitude;
blue-grey-orange diverging scale for correlation. Every chart has a title, labelled axes/legends and a one-line "Insight" caption.


