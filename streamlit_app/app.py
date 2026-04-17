import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Severity Prediction", layout="wide")

st.markdown("""
<style>
  /* Force white background everywhere */
  .stApp { background-color: #ffffff !important; }
  .main { background-color: #ffffff !important; }
  .block-container { background-color: #ffffff !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #f5f5f5 !important; }

  /* ALL text black */
  * { color: #000000 !important; font-family: "Times New Roman", Times, serif !important; }

  /* Dropdown/selectbox popup background */
  ul[data-testid="stSelectboxVirtualDropdown"],
  div[data-baseweb="popover"],
  div[data-baseweb="menu"],
  ul[role="listbox"],
  li[role="option"] {
      background-color: #ffffff !important;
      color: #000000 !important;
  }
  li[role="option"]:hover {
      background-color: #f0f0f0 !important;
  }

  /* Selectbox input box */
  div[data-baseweb="select"] > div {
      background-color: #ffffff !important;
      border: 1px solid #cccccc !important;
      color: #000000 !important;
  }

  /* Input text inside selectbox */
  div[data-baseweb="select"] input {
      background-color: #ffffff !important;
      color: #000000 !important;
  }

  /* Metric cards */
  [data-testid="stMetricDelta"] { display: none !important; }
  [data-testid="stMetric"] { background-color: #f0f0f0 !important; border-radius: 8px; padding: 12px; }

  /* Radio buttons */
  input[type="radio"] { accent-color: #1f77b4 !important; }

  /* Slider */
  div[data-testid="stSlider"] input[type="range"] { accent-color: #1f77b4 !important; }
  div[data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none !important;
      background-color: #1f77b4 !important;
      width: 18px !important; height: 18px !important;
      border-radius: 50% !important; cursor: pointer !important;
  }
  div[data-testid="stSlider"] input[type="range"]::-webkit-slider-runnable-track {
      background-color: #cccccc !important; height: 4px !important;
  }

  /* Links */
  a { color: #1a0dab !important; }

  /* Code blocks */
  code { background-color: #e8e8e8 !important; color: #000000 !important; padding: 2px 5px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/parkinsons_updrs.data.csv")

data = load_data()

page = st.sidebar.selectbox("Navigate", ["Landing Page", "Interactive Visualizations", "MLP Demo"])

def clean_layout(fig, title=None, xaxis_title=None, yaxis_title=None, yaxis_range=None):
    layout = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman", color="black"),
        title_font=dict(family="Times New Roman", color="black"),
        legend=dict(font=dict(color="black")),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(color="black", family="Times New Roman"))
    xax = dict(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", gridcolor="#eeeeee")
    if xaxis_title:
        xax["title"] = xaxis_title
    yax = dict(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", gridcolor="#eeeeee")
    if yaxis_title:
        yax["title"] = yaxis_title
    if yaxis_range:
        yax["range"] = yaxis_range
    layout["xaxis"] = xax
    layout["yaxis"] = yax
    fig.update_layout(**layout)
    return fig

# ============================================================
# PAGE 1: LANDING PAGE
# ============================================================
if page == "Landing Page":
    st.title("Predicting Parkinson's Disease Severity from Voice")
    st.markdown("### DS 4420 Final Project — Spring 2026")
    st.markdown("**Rhea Wadhwa and Rhoda Zerit**")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
## Project Overview
Parkinson's disease is a progressive neurological disorder affecting millions worldwide.
As the disease progresses, it causes changes in motor control — including the voice.
This project uses **speech biomarkers** (acoustic features derived from voice recordings)
to predict the **motor UPDRS score**, a clinical measure of Parkinson's severity.

## Why Voice?
Voice recordings are non-invasive, inexpensive, and can be collected remotely.
If we can reliably predict motor severity from voice alone, this could enable
remote patient monitoring and earlier detection of disease progression.

## Our Approach
We trained and compared three machine learning models:
        """)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Bayesian Linear (Manual)", "R² = 0.30")
        with col_b:
            st.metric("Bayesian (brms)", "R² = 0.17")
        with col_c:
            st.metric("MLP Neural Network", "R² = 0.69")

        st.markdown("""
## Key Finding
Linear Bayesian models are interpretable but hit a ceiling due to the nonlinear
relationship between voice features and motor severity. A manually-implemented
MLP neural network significantly outperformed both Bayesian models,
achieving an R² of **0.69** — demonstrating the power of nonlinear modeling for this task.
        """)

    with col2:
        st.markdown("## Dataset")
        st.markdown("""
**Parkinson's Telemonitoring Dataset**
- **Source:** UCI Machine Learning Repository
- **Patients:** 42 subjects
- **Recordings:** 5,875 voice samples
- **Features:** 16 acoustic features
- **Target:** motor UPDRS score (0–108)

[Dataset Link](https://www.kaggle.com/datasets/thedevastator/unlocking-clues-to-parkinson-s-disease-progressi/data)
        """)
        st.markdown("## Features Used")
        for f in ["Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE", "age", "sex", "test_time"]:
            st.markdown(f"- {f}")
        st.markdown("## Methods")
        st.markdown("""
- Bayesian Linear Regression (manual, R)
- Bayesian Linear Regression (brms, R)
- MLP Neural Network (manual, Python)
        """)

# ============================================================
# PAGE 2: INTERACTIVE VISUALIZATIONS
# ============================================================
elif page == "Interactive Visualizations":
    st.title("Interactive Data Visualizations")
    st.markdown("Explore the Parkinson's dataset and model results interactively.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Feature Explorer", "Patient Progression", "Model Comparison"])

    with tab1:
        st.subheader("Voice Feature vs Motor UPDRS")
        st.markdown("Select a voice feature to see its relationship with motor severity.")

        feature_cols = [c for c in data.columns if c not in ["motor_UPDRS", "total_UPDRS", "subject#", "index"]]
        default_idx = feature_cols.index("PPE") if "PPE" in feature_cols else 0
        selected_feature = st.selectbox("Choose a feature:", feature_cols, index=default_idx)
        color_by = st.radio("Color points by:", ["motor_UPDRS", "age", "sex"], horizontal=True)

        x_vals = data[selected_feature].values.astype(float)
        y_vals = data["motor_UPDRS"].values.astype(float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        m, b = np.polyfit(x_vals[mask], y_vals[mask], 1)
        x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
        y_line = m * x_line + b

        fig = px.scatter(
            data, x=selected_feature, y="motor_UPDRS",
            color=color_by, opacity=0.4,
            color_continuous_scale="Viridis",
            labels={"motor_UPDRS": "Motor UPDRS Score"},
        )
        fig.update_traces(marker=dict(size=4))
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=False
        ))
        fig = clean_layout(fig, title=f"{selected_feature} vs Motor UPDRS",
                           xaxis_title=selected_feature, yaxis_title="Motor UPDRS Score")
        fig.update_coloraxes(colorbar=dict(tickfont=dict(color="black"), title=dict(font=dict(color="black"))))
        st.plotly_chart(fig, use_container_width=True)

        corr = data[selected_feature].corr(data["motor_UPDRS"])
        st.info(f"Pearson correlation between **{selected_feature}** and **motor_UPDRS**: {corr:.4f}")

    with tab2:
        st.subheader("Individual Patient Progression Over Time")
        st.markdown("Track how a patient's motor UPDRS score changes across recording sessions.")

        subject_ids = sorted(data["subject#"].unique())
        selected_subject = st.selectbox("Select a patient (subject #):", subject_ids)
        patient_df = data[data["subject#"] == selected_subject].sort_values("test_time")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=patient_df["test_time"], y=patient_df["motor_UPDRS"],
            mode="lines+markers", name="motor UPDRS",
            line=dict(color="royalblue", width=2), marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=patient_df["test_time"], y=patient_df["total_UPDRS"],
            mode="lines+markers", name="total UPDRS",
            line=dict(color="tomato", width=2, dash="dash"), marker=dict(size=6)
        ))
        fig2 = clean_layout(fig2, title=f"Patient {selected_subject}: UPDRS Over Time",
                            xaxis_title="Test Time (days)", yaxis_title="UPDRS Score")
        fig2.update_layout(hovermode="x unified", legend=dict(orientation="h", font=dict(color="black")))
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Recordings", len(patient_df))
        c2.metric("Avg Motor UPDRS", f"{patient_df['motor_UPDRS'].mean():.1f}")
        c3.metric("Age", int(patient_df['age'].iloc[0]))

    with tab3:
        st.subheader("Model Performance Comparison")
        models = ["Bayesian (brms)", "Bayesian (Manual)", "MLP"]
        r2_scores = [0.166, 0.30, 0.69]
        rmse_scores = [7.21, 6.79, 4.42]
        colors = ["#4C72B0", "#DD8452", "#55A868"]

        col1, col2 = st.columns(2)
        with col1:
            fig3 = go.Figure(go.Bar(
                x=models, y=r2_scores, marker_color=colors,
                text=[f"{v:.3f}" for v in r2_scores], textposition="outside",
                textfont=dict(color="black", family="Times New Roman")
            ))
            fig3 = clean_layout(fig3, title="R² Score by Model (higher = better)",
                                xaxis_title="Model", yaxis_title="R²", yaxis_range=[0, 0.85])
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = go.Figure(go.Bar(
                x=models, y=rmse_scores, marker_color=colors,
                text=[f"{v:.2f}" for v in rmse_scores], textposition="outside",
                textfont=dict(color="black", family="Times New Roman")
            ))
            fig4 = clean_layout(fig4, title="RMSE by Model (lower = better)",
                                xaxis_title="Model", yaxis_title="RMSE", yaxis_range=[0, 9])
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""
The MLP dramatically outperforms both linear Bayesian models (R² = 0.698 vs. 0.30 and 0.166),
highlighting that the relationship between voice features and motor UPDRS is fundamentally nonlinear.
The Bayesian models, while weaker predictively, offer interpretability through posterior distributions on coefficients.
        """)

# ============================================================
# PAGE 3: MLP DEMO
# ============================================================
elif page == "MLP Demo":
    st.title("Live MLP Prediction Demo")
    st.markdown("Adjust voice feature values below and get a predicted motor UPDRS score from our trained MLP.")
    st.divider()

    @st.cache_resource
    def train_model():
        np.random.seed(42)
        drop_cols = [c for c in ["motor_UPDRS", "total_UPDRS", "subject#", "index"] if c in data.columns]
        X = data.drop(columns=drop_cols).values.astype(float)
        y = data["motor_UPDRS"].values.astype(float).reshape(-1, 1)
        n = X.shape[0]
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.8 * n)]
        X_tr = X[train_idx]; y_tr = y[train_idx]
        Xm = X_tr.mean(axis=0); Xs = X_tr.std(axis=0); Xs[Xs == 0] = 1
        X_train = (X_tr - Xm) / Xs
        ym = float(y_tr.mean()); ys = float(y_tr.std())
        y_train = (y_tr - ym) / ys
        d = X_train.shape[1]; h1, h2 = 64, 32; eta = 0.01
        W1 = np.random.randn(d, h1) * np.sqrt(2/d); b1 = np.zeros((1, h1))
        W2 = np.random.randn(h1, h2) * np.sqrt(2/h1); b2 = np.zeros((1, h2))
        W3 = np.random.randn(h2, 1) * np.sqrt(2/h2); b3 = np.zeros((1, 1))
        for epoch in range(500):
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), 64):
                Xb = X_train[perm[i:i+64]]; yb = y_train[perm[i:i+64]]
                Z1 = Xb @ W1 + b1; H1 = np.maximum(0, Z1)
                Z2 = H1 @ W2 + b2; H2 = np.tanh(Z2)
                yhat = H2 @ W3 + b3
                dy = (2/len(Xb)) * (yhat - yb)
                dW3 = H2.T @ dy; db3 = dy.sum(0, keepdims=True)
                dH2 = dy @ W3.T; dZ2 = dH2 * (1 - np.tanh(Z2)**2)
                dW2 = H1.T @ dZ2; db2 = dZ2.sum(0, keepdims=True)
                dH1 = dZ2 @ W2.T; dZ1 = dH1 * (Z1 > 0)
                dW1 = Xb.T @ dZ1; db1 = dZ1.sum(0, keepdims=True)
                W3 -= eta*dW3; b3 -= eta*db3
                W2 -= eta*dW2; b2 -= eta*db2
                W1 -= eta*dW1; b1 -= eta*db1
            eta *= 0.995
        feat_names = [c for c in data.columns if c not in ["motor_UPDRS", "total_UPDRS", "subject#", "index"]]
        return (W1, b1, W2, b2, W3, b3), Xm, Xs, ym, ys, feat_names

    with st.spinner("Training MLP model — please wait about 10 seconds..."):
        weights, X_mean, X_std, y_mean, y_std, feature_names = train_model()

    W1, b1, W2, b2, W3, b3 = weights

    def predict_one(x_raw):
        x = (x_raw - X_mean) / X_std
        H1 = np.maximum(0, x @ W1 + b1)
        H2 = np.tanh(H1 @ W2 + b2)
        return float(np.squeeze(H2 @ W3 + b3)) * y_std + y_mean

    st.markdown("### Adjust Feature Values")
    st.markdown("Use the sliders to set feature values. All other features are held at their median values.")

    defaults = data[feature_names].median()
    col1, col2 = st.columns(2)
    user_inputs = {}
    key_features = [f for f in ["age", "sex", "test_time", "Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE"] if f in feature_names]
    other_features = [f for f in feature_names if f not in key_features]

    for i, feat in enumerate(key_features):
        col = col1 if i % 2 == 0 else col2
        mn, mx, default = float(data[feat].min()), float(data[feat].max()), float(defaults[feat])
        user_inputs[feat] = col.slider(feat, min_value=mn, max_value=mx, value=default, key=feat)
    for feat in other_features:
        user_inputs[feat] = float(defaults[feat])

    x_input = np.array([user_inputs[f] for f in feature_names], dtype=float).reshape(1, -1)
    prediction = float(np.clip(predict_one(x_input), 0, 108))

    st.divider()
    st.markdown("### Prediction")
    col_pred, col_gauge = st.columns([1, 2])

    with col_pred:
        st.metric("Predicted Motor UPDRS", f"{prediction:.1f}")
        severity = "Mild" if prediction < 20 else "Moderate" if prediction < 35 else "Severe"
        st.markdown(f"**Severity Category:** {severity}")

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Motor UPDRS Score", "font": {"family": "Times New Roman", "color": "black"}},
            number={"font": {"color": "black", "family": "Times New Roman"}},
            gauge={
                "axis": {"range": [0, 108], "tickfont": {"color": "black"}},
                "bar": {"color": "steelblue"},
                "steps": [
                    {"range": [0, 20], "color": "#d4edda"},
                    {"range": [20, 35], "color": "#fff3cd"},
                    {"range": [35, 108], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": prediction}
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=0),
                                paper_bgcolor="white",
                                font=dict(family="Times New Roman", color="black"))
        st.plotly_chart(fig_gauge, use_container_width=True)
