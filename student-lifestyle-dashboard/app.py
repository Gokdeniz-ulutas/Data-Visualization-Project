"""Student Lifestyle & Academic Performance - interactive Streamlit dashboard.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ml_models import (
    CLUSTER_FEATURES,
    REGRESSION_FEATURES,
    fit_gpa_regression,
    kmeans_elbow,
    run_kmeans,
)
from preprocess import CLEAN_PATH, HOUR_COLS, NUMERIC_COLS, STRESS_ORDER, load_and_clean

st.set_page_config(page_title="Student Lifestyle Dashboard", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------------------
# Visual identity: one colour per entity, used consistently in every chart
# --------------------------------------------------------------------------------------
STRESS_COLORS = {"Low": "#1baf7a", "Moderate": "#2a78d6", "High": "#eb6834"}
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]  # clusters / bands, fixed order
SEQ = ["#dbe8f9", "#2a78d6", "#0d3b78"]  # sequential (magnitude), single hue
DIV = [[0.0, "#2a78d6"], [0.5, "#f2f2f2"], [1.0, "#eb6834"]]  # diverging (correlation)
INK, INK_SOFT = "#0b0b0b", "#52514e"

SHORT = {
    "Study_Hours_Per_Day": "Study",
    "Extracurricular_Hours_Per_Day": "Extracurricular",
    "Sleep_Hours_Per_Day": "Sleep",
    "Social_Hours_Per_Day": "Social",
    "Physical_Activity_Hours_Per_Day": "Physical",
    "GPA": "GPA",
    "Stress_Code": "Stress (0-2)",
}
LONG = {
    "Study hours": "Study_Hours_Per_Day",
    "Sleep hours": "Sleep_Hours_Per_Day",
    "Social hours": "Social_Hours_Per_Day",
    "Extracurricular hours": "Extracurricular_Hours_Per_Day",
    "Physical activity hours": "Physical_Activity_Hours_Per_Day",
    "GPA": "GPA",
}


def style(fig, height=470):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=13, color=INK),
        title_font=dict(size=16, color=INK),
        legend=dict(font=dict(color=INK_SOFT)),
    )
    return fig


def show(fig):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def hex_to_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@st.cache_data
def get_data():
    # The cleaning pipeline lives in preprocess.py; it is re-run here so the app is self-contained.
    return load_and_clean()


@st.cache_data
def cached_kmeans(d, k):
    return run_kmeans(d, k)


@st.cache_data
def cached_elbow(d):
    return kmeans_elbow(d)


df, quality = get_data()

# --------------------------------------------------------------------------------------
# Sidebar: interactive filters (they drive every chart)
# --------------------------------------------------------------------------------------
st.sidebar.header("🎛️ Filters")
stress_sel = st.sidebar.multiselect("Stress level", STRESS_ORDER, default=STRESS_ORDER)
study_rng = st.sidebar.slider("Study hours / day", 5.0, 10.0, (5.0, 10.0), step=0.1)
sleep_rng = st.sidebar.slider("Sleep hours / day", 5.0, 10.0, (5.0, 10.0), step=0.1)
gpa_rng = st.sidebar.slider("GPA", 2.0, 4.0, (2.0, 4.0), step=0.05)
exclude_out = st.sidebar.checkbox(
    f"Exclude flagged outliers ({quality['outlier_rows']} students)", value=False,
    help="Outliers were flagged with the 1.5 x IQR rule during preprocessing.",
)
st.sidebar.markdown("---")
st.sidebar.header("🤖 ML settings")
k_clusters = st.sidebar.slider("Number of K-Means clusters (k)", 2, 5, 3)

mask = (
    df["Stress_Level"].isin(stress_sel)
    & df["Study_Hours_Per_Day"].between(*study_rng)
    & df["Sleep_Hours_Per_Day"].between(*sleep_rng)
    & df["GPA"].between(*gpa_rng)
)
if exclude_out:
    mask &= ~df["Is_Outlier"]
d = df[mask].copy()

st.title("🎓 Student Lifestyle & Academic Performance")
st.caption(
    "Daily time use, stress and GPA of 2,000 students. Use the sidebar filters - every chart updates. "
    "Hover for details, drag to zoom/pan, click legend items to isolate groups."
)

if len(d) < 30:
    st.warning("Fewer than 30 students match the current filters. Please widen the filters.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Students shown", f"{len(d):,}", f"{len(d) / len(df):.0%} of all")
c2.metric("Average GPA", f"{d['GPA'].mean():.2f}")
c3.metric("Study hours / day", f"{d['Study_Hours_Per_Day'].mean():.1f}")
c4.metric("Sleep hours / day", f"{d['Sleep_Hours_Per_Day'].mean():.1f}")
c5.metric("High-stress share", f"{(d['Stress_Level'] == 'High').mean():.0%}")

tab1, tab2, tab3, lecture_tab, tab4 = st.tabs(
    ["📊 1 · Relationships & distributions", "🧭 2 · Multi-dimensional views", "🤖 3 · Machine learning", "Lecture charts", "🧹 Data & quality"]
)

# ======================================================================================
# TAB 1 - Member 1: relationships and distributions
# ======================================================================================
with tab1:
    # ---- Viz 1: correlation heatmap -------------------------------------------------
    st.subheader("Viz 1 · Which variables move together?")
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, key="corr_method")
    corr_cols = NUMERIC_COLS + ["Stress_Code"]
    corr = d[corr_cols].corr(method=method).rename(index=SHORT, columns=SHORT)
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=DIV, aspect="auto",
        title=f"Correlation matrix ({method.title()}) - blue = negative, orange = positive",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    show(style(fig, 480))
    gpa_corr = corr["GPA"].drop(["GPA"])
    top = gpa_corr.abs().idxmax()
    st.caption(
        f"**Insight:** {top} time has the strongest link with GPA (r = {gpa_corr[top]:+.2f}). "
        f"Physical activity correlates negatively with study time because the five daily activities always add up to 24 h "
        f"(r = {corr.loc['Physical', 'Study']:+.2f})."
    )

    st.markdown("---")
    # ---- Viz 2: violin by stress level ----------------------------------------------
    st.subheader("Viz 2 · How does each metric differ by stress level?")
    metric_label = st.selectbox("Metric to compare", list(LONG.keys()), index=5, key="violin_metric")
    ycol = LONG[metric_label]
    fig = px.violin(
        d, x="Stress_Level", y=ycol, color="Stress_Level", box=True, points="outliers",
        category_orders={"Stress_Level": STRESS_ORDER}, color_discrete_map=STRESS_COLORS,
        hover_data=["Student_ID"], title=f"{metric_label} by stress level (violin + box plot)",
        labels={"Stress_Level": "Stress level", ycol: metric_label},
    )
    fig.update_layout(showlegend=False)
    show(style(fig, 440))
    means = d.groupby("Stress_Level", observed=True)[ycol].mean()
    parts = ", ".join(f"{k}: {v:.2f}" for k, v in means.items())
    st.caption(f"**Insight:** mean {metric_label.lower()} by stress level - {parts}.")

    st.markdown("---")
    # ---- Viz 3: density heatmap with marginals ---------------------------------------
    st.subheader("Viz 3 · Where do students concentrate?")
    xlabel = st.selectbox("X-axis variable", [k for k in LONG if k != "GPA"], index=0, key="density_x")
    xcol = LONG[xlabel]
    fig = px.density_heatmap(
        d, x=xcol, y="GPA", nbinsx=20, nbinsy=20, marginal_x="histogram", marginal_y="histogram",
        color_continuous_scale=SEQ, labels={xcol: xlabel}, title=f"Density of students: {xlabel} vs GPA (with marginal histograms)",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Students"))
    show(style(fig, 520))
    r = d[[xcol, "GPA"]].corr().iloc[0, 1]
    st.caption(f"**Insight:** darker cells hold more students. Correlation between {xlabel.lower()} and GPA is r = {r:+.2f}.")

# ======================================================================================
# TAB 2 - Member 2: multi-dimensional views
# ======================================================================================
with tab2:
    # ---- Viz 4: treemap --------------------------------------------------------------
    st.subheader("Viz 4 · How are students split by stress, study time and GPA?")
    depth = st.radio("Hierarchy depth", ["Stress → Study band", "Stress → Study band → GPA band"], horizontal=True, key="tm_depth")
    g = (
        d.assign(Stress=d["Stress_Level"].astype(str), Study=d["Study_Band"].astype(str), GPA_Group=d["GPA_Band"].astype(str))
        .groupby(["Stress", "Study", "GPA_Group"])
        .agg(Students=("GPA", "size"), Avg_GPA=("GPA", "mean"))
        .reset_index()
    )
    path = [px.Constant("All students"), "Stress", "Study"]
    if "GPA band" in depth:
        path.append("GPA_Group")
    fig = px.treemap(
        g, path=path, values="Students", color="Avg_GPA", color_continuous_scale=SEQ,
        labels={"Avg_GPA": "Avg GPA"}, title="Treemap: area = number of students, colour = average GPA",
    )
    fig.update_traces(root_color="#f2f2f2", marker=dict(line=dict(width=2, color="white")))
    fig.update_layout(coloraxis_colorbar=dict(title="Avg GPA"))
    show(style(fig, 500))
    high = d[d["Stress_Level"] == "High"]
    share_high_study = (high["Study_Band"].astype(str) == "High (>8h)").mean() if len(high) else float("nan")
    st.caption(
        f"**Insight:** darker blocks = higher GPA. Among high-stress students, {share_high_study:.0%} study more than 8 h a day. "
        "Click any block to zoom into it."
    )

    st.markdown("---")
    # ---- Viz 5: parallel coordinates ---------------------------------------------------
    st.subheader("Viz 5 · Full lifestyle profile of every student")
    pc = d[HOUR_COLS + ["Stress_Code", "GPA"]]
    fig = px.parallel_coordinates(
        pc, dimensions=list(pc.columns), color="GPA", color_continuous_scale=SEQ,
        labels={**SHORT, "Stress_Code": "Stress (0 Low, 2 High)"},
        title="Parallel coordinates - each line is one student, coloured by GPA",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="GPA"), margin=dict(l=60, r=40, t=80, b=20))
    show(style(fig, 470))
    st.caption(
        "**Insight:** drag along any axis to brush a range (e.g. sleep < 6 h) and see which GPA colours remain. "
        "High-GPA (dark) lines cluster at high study hours and low physical activity."
    )

    st.markdown("---")
    # ---- Viz 6: Sankey ---------------------------------------------------------------
    st.subheader("Viz 6 · Flow from study habits to GPA")
    stages = {"Study_Band": "Study time", "Sleep_Band": "Sleep", "Stress_Level": "Stress", "GPA_Band": "GPA"}
    chosen = st.multiselect(
        "Stages to show (in order)", list(stages.keys()), default=list(stages.keys()), format_func=lambda c: stages[c], key="sankey_stages"
    )
    if len(chosen) < 2:
        st.info("Select at least two stages.")
    else:
        stage_colors = {"Study_Band": "#2a78d6", "Sleep_Band": "#6250d6", "GPA_Band": "#52514e"}
        labels, node_color, index = [], [], {}
        for col in chosen:
            for cat in [str(c) for c in d[col].cat.categories if (d[col] == c).any()]:
                index[(col, cat)] = len(labels)
                labels.append(cat)
                node_color.append(STRESS_COLORS[cat] if col == "Stress_Level" else stage_colors[col])
        src, tgt, val = [], [], []
        for a, b in zip(chosen[:-1], chosen[1:]):
            pairs = d.groupby([a, b], observed=True).size().reset_index(name="n")
            for _, row in pairs.iterrows():
                if row["n"] > 0:
                    src.append(index[(a, str(row[a]))])
                    tgt.append(index[(b, str(row[b]))])
                    val.append(int(row["n"]))
        fig = go.Figure(
            go.Sankey(
                node=dict(label=labels, color=node_color, pad=18, thickness=18, line=dict(color="white", width=1)),
                link=dict(source=src, target=tgt, value=val, color=[hex_to_rgba(node_color[s], 0.30) for s in src]),
            )
        )
        fig.update_layout(title="Sankey diagram: how groups of students flow through the chosen stages")
        for i, col in enumerate(chosen):
            fig.add_annotation(
                x=i / (len(chosen) - 1), y=1.07, xref="paper", yref="paper", text=f"<b>{stages[col]}</b>",
                showarrow=False, font=dict(color=INK_SOFT, size=13),
            )
        show(style(fig, 480))
        st.caption("**Insight:** band thickness = number of students. Hover a link to see how many students follow that path.")

# ======================================================================================
# TAB 3 - Member 3: machine learning
# ======================================================================================
with tab3:
    if len(d) < 60:
        st.warning("Machine-learning views need at least 60 students. Please widen the filters.")
    else:
        labels, sil, profile = cached_kmeans(d, k_clusters)
        d_ml = d.assign(Cluster=labels)
        cluster_names = sorted(d_ml["Cluster"].unique())
        cluster_color = {c: GROUP_COLORS[i % len(GROUP_COLORS)] for i, c in enumerate(cluster_names)}
        symbols = ["circle", "diamond", "square", "cross", "x"]

        # ---- Viz 7: 3D scatter coloured by K-Means cluster ---------------------------
        st.subheader("Viz 7 · K-Means clusters of students (ML algorithm 1: clustering)")
        a, b, c = st.columns(3)
        xl = a.selectbox("X axis", list(LONG.keys()), index=0, key="x3d")
        yl = b.selectbox("Y axis", list(LONG.keys()), index=1, key="y3d")
        zl = c.selectbox("Z axis", list(LONG.keys()), index=5, key="z3d")
        fig = px.scatter_3d(
            d_ml, x=LONG[xl], y=LONG[yl], z=LONG[zl], color="Cluster", symbol="Cluster",
            color_discrete_map=cluster_color, symbol_sequence=symbols, opacity=0.75,
            category_orders={"Cluster": cluster_names}, hover_data=["Student_ID", "GPA", "Stress_Level"],
            labels={LONG[xl]: xl, LONG[yl]: yl, LONG[zl]: zl},
            title=f"3-D view of K-Means clusters (k = {k_clusters}; clusters ordered by average GPA)",
        )
        fig.update_traces(marker=dict(size=3.5))
        show(style(fig, 560))
        m1, m2 = st.columns([1, 3])
        m1.metric("Silhouette score", f"{sil:.2f}", help="1 = perfectly separated clusters, 0 = overlapping clusters.")
        with m2:
            show_df(profile.rename(columns=SHORT))
        st.caption(
            f"**Insight:** clustering on all six variables separates a high-GPA, heavy-study group from the rest. "
            f"The silhouette score of {sil:.2f} shows the groups overlap - lifestyle is more of a continuum than clear-cut types."
        )
        with st.expander("How was k chosen? (elbow & silhouette table)"):
            show_df(cached_elbow(d).round(3))

        st.markdown("---")
        # ---- Viz 8: radar chart of group profiles ------------------------------------
        st.subheader("Viz 8 · Profile of each group (radar chart)")
        group_by = st.selectbox("Compare groups by", ["Cluster", "Stress_Level", "Study_Band"], key="radar_group")
        radar_cols = HOUR_COLS + ["GPA"]
        overall = df[radar_cols].mean()
        order = cluster_names if group_by == "Cluster" else (
            STRESS_ORDER if group_by == "Stress_Level" else [str(x) for x in d_ml["Study_Band"].cat.categories]
        )
        colors = cluster_color if group_by == "Cluster" else (
            STRESS_COLORS if group_by == "Stress_Level" else {o: GROUP_COLORS[i] for i, o in enumerate(order)}
        )
        gmeans = d_ml.assign(G=d_ml[group_by].astype(str)).groupby("G")[radar_cols].mean()
        theta = [SHORT[c] for c in radar_cols]
        fig = go.Figure()
        all_idx = []
        for grp in [o for o in order if o in gmeans.index]:
            actual = gmeans.loc[grp]
            idx = (actual / overall * 100).values
            all_idx.extend(idx)
            fig.add_trace(
                go.Scatterpolar(
                    r=list(idx) + [idx[0]], theta=theta + [theta[0]], name=grp, fill="toself", opacity=0.55,
                    line=dict(color=colors[grp], width=2), fillcolor=hex_to_rgba(colors[grp], 0.15),
                    customdata=list(actual.values) + [actual.values[0]],
                    hovertemplate="%{theta}<br>Index: %{r:.0f}<br>Average: %{customdata:.2f}<extra>" + grp + "</extra>",
                )
            )
        lo = max(0, int(np.floor((min(all_idx) - 10) / 10) * 10))
        hi = int(np.ceil((max(all_idx) + 10) / 10) * 10)
        fig.update_layout(
            title="Radar: average of each metric as an index (100 = average of all 2,000 students)",
            polar=dict(radialaxis=dict(range=[lo, hi], ticksuffix="", gridcolor="#e6e6e3")),
        )
        show(style(fig, 520))
        st.caption(
            "**Insight:** an index above 100 means the group spends more time on that activity (or has a higher GPA) than the average student. "
            "Click legend entries to compare groups one by one."
        )

        st.markdown("---")
        # ---- Viz 9: linear regression ----------------------------------------------------
        st.subheader("Viz 9 · Predicting GPA (ML algorithm 2: linear regression)")
        res = fit_gpa_regression(d)
        r1, r2, r3 = st.columns(3)
        r1.metric("R² (test set)", f"{res['r2']:.2f}")
        r2.metric("RMSE (GPA points)", f"{res['rmse']:.2f}")
        r3.metric("Train / test students", f"{res['n_train']} / {res['n_test']}")
        test = res["test"].join(d["Stress_Level"])
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.62, 0.38],
            subplot_titles=("Actual vs predicted GPA (held-out test set)", "Standardised coefficients (effect of +1 SD)"),
        )
        for s in STRESS_ORDER:
            sub = test[test["Stress_Level"] == s]
            if len(sub):
                fig.add_trace(
                    go.Scatter(
                        x=sub["Actual"], y=sub["Predicted"], mode="markers", name=f"{s} stress",
                        marker=dict(size=6, color=STRESS_COLORS[s], opacity=0.65, line=dict(width=0.5, color="white")),
                        hovertemplate="Actual %{x:.2f}<br>Predicted %{y:.2f}<extra>" + s + "</extra>",
                    ),
                    row=1, col=1,
                )
        lo_, hi_ = float(test[["Actual", "Predicted"]].min().min()), float(test[["Actual", "Predicted"]].max().max())
        fig.add_trace(
            go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", name="Perfect prediction", line=dict(color=INK_SOFT, dash="dash", width=1.5)),
            row=1, col=1,
        )
        coef = res["std_coef"]
        fig.add_trace(
            go.Bar(
                x=coef.values, y=[SHORT[i] for i in coef.index], orientation="h", showlegend=False,
                marker_color=["#2a78d6" if v >= 0 else "#eb6834" for v in coef.values],
                hovertemplate="%{y}: %{x:+.3f} GPA points per +1 SD<extra></extra>",
            ),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="Actual GPA", row=1, col=1)
        fig.update_yaxes(title_text="Predicted GPA", row=1, col=1)
        fig.update_xaxes(title_text="GPA change per +1 SD", row=1, col=2)
        show(style(fig, 470))
        best = res["std_coef"].abs().idxmax()
        st.caption(
            f"**Insight:** the model explains {res['r2']:.0%} of GPA variation on unseen students. "
            f"{SHORT[best]} time is the dominant predictor - one extra hour changes predicted GPA by {res['coef'][best]:+.2f} points. "
            "Physical activity is left out of the model because the five activities sum to 24 h (perfect collinearity)."
        )

        with st.expander("🔮 What-if GPA predictor (interactive)"):
            w1, w2, w3, w4 = st.columns(4)
            vals = {
                "Study_Hours_Per_Day": w1.slider("Study h/day", 5.0, 10.0, 7.0, 0.1, key="wi_study"),
                "Sleep_Hours_Per_Day": w2.slider("Sleep h/day", 5.0, 10.0, 7.5, 0.1, key="wi_sleep"),
                "Social_Hours_Per_Day": w3.slider("Social h/day", 0.0, 6.0, 2.7, 0.1, key="wi_social"),
                "Extracurricular_Hours_Per_Day": w4.slider("Extracurricular h/day", 0.0, 4.0, 2.0, 0.1, key="wi_extra"),
            }
            pred = float(res["model"].predict(pd.DataFrame([vals], columns=REGRESSION_FEATURES))[0])
            st.metric("Predicted GPA", f"{min(max(pred, 0), 4):.2f}")
            used = sum(vals.values())
            st.caption(f"These four activities use {used:.1f} h of the 24-hour day (the rest is physical activity).")

# ======================================================================================
# TAB 4 - data and quality
# ======================================================================================
with lecture_tab:
    st.subheader("Viz 10 · GPA distribution")
    bin_count = st.slider("Approximate number of GPA bins", 10, 40, 20)
    show(style(px.histogram(
        d, x="GPA", nbins=bin_count, title="How common is each GPA range?",
        labels={"count": "Students"}, color_discrete_sequence=[GROUP_COLORS[0]],
    ).update_yaxes(title_text="Students")))
    st.caption("DataVis 4, slide 12. Change the bin count to see how bin width affects the shape.")

    st.subheader("Viz 11 · Pairwise time-use relationships")
    pairs = ["Study_Hours_Per_Day", "Sleep_Hours_Per_Day",
             "Social_Hours_Per_Day", "Physical_Activity_Hours_Per_Day"]
    matrix = px.scatter_matrix(
        d, dimensions=pairs, color="Stress_Level", color_discrete_map=STRESS_COLORS,
        category_orders={"Stress_Level": STRESS_ORDER}, labels=SHORT,
        hover_data=["GPA"], title="Study, sleep, social and physical activity hours per day",
    )
    matrix.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=.45))
    show(style(matrix, 650))
    st.caption("DataVis 5, slide 31. Each panel compares two activities. Click the legend to isolate a stress group.")

    st.subheader("Viz 12 · Stress and sleep groups")
    hierarchy = d.assign(
        Sleep_band=pd.cut(d["Sleep_Hours_Per_Day"], [-float("inf"), 7, 8, float("inf")],
                          labels=["Under 7 h", "7 to under 8 h", "8 h or more"], right=False),
        Study_band=pd.cut(d["Study_Hours_Per_Day"], [-float("inf"), 7, 9, float("inf")],
                          labels=["Under 7 h study", "7 to under 9 h study", "9 h study or more"], right=False),
    )
    groups = hierarchy.groupby(
        ["Stress_Level", "Sleep_band", "Study_band"], observed=True, as_index=False
    ).agg(Students=("Student_ID", "size"))
    for column in ["Stress_Level", "Sleep_band", "Study_band"]:
        groups[column] = groups[column].astype(str)
    icicle = px.icicle(
        groups, path=[px.Constant("All selected students"), "Stress_Level", "Sleep_band", "Study_band"],
        values="Students", color="Stress_Level", color_discrete_map=STRESS_COLORS,
        title="How are sleep and study bands split within each stress group?",
    )
    show(style(icicle))
    st.caption("DataVis 6, slide 15. Width represents student count. Click a stress or sleep block to drill down.")

with tab4:
    st.subheader("Dataset")
    st.write(
        "**Source:** Student Lifestyle dataset (Kaggle) · **Rows:** 2,000 · **Original columns:** 8 "
        "(1 identifier, 5 numerical hour columns, GPA, categorical stress level). "
        "Cleaning added 6 derived columns (total hours, outlier flag, stress code and three categorical bands)."
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing values", quality["missing_before"])
    q2.metric("Duplicate students", quality["duplicate_rows"])
    q3.metric("Days not summing to 24 h", quality["days_not_24h"])
    q4.metric("Outlier rows (IQR)", quality["outlier_rows"])
    st.markdown("**Outliers flagged per column (1.5 × IQR rule)** - flagged rows are kept by default; tick the sidebar box to exclude them.")
    out = pd.DataFrame(
        {
            "Outliers": pd.Series(quality["outliers_per_column"]),
            "Lower bound": pd.Series({k: v[0] for k, v in quality["iqr_bounds"].items()}),
            "Upper bound": pd.Series({k: v[1] for k, v in quality["iqr_bounds"].items()}),
        }
    )
    show_df(out)
    st.markdown("**Filtered data preview**")
    show_df(d.drop(columns=["Stress_Code"]).head(200))
    st.download_button("⬇️ Download filtered data (CSV)", d.to_csv(index=False).encode("utf-8"), "student_lifestyle_filtered.csv", "text/csv")
