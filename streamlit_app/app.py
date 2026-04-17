import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Severity Prediction", layout="wide")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/parkinsons_updrs.data.csv")

data = load_data()

# ----------------------------
# Sidebar navigation
# ----------------------------
page = st.sidebar.selectbox("Navigate", ["🏠 Landing Page", "📊 Interactive Visualizations", "🤖 MLP Demo"])

# ============================================================
# PAGE 1: LANDING PAGE
# ============================================================
if page == "🏠 Landing Page":
    st.title("🧠 Predicting Parkinson's Disease Severity from Voice")
    st.markdown("### DS 4420 Final Project — Spring 2026")
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
        We trained and compared **three machine learning models**:
        """)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Bayesian Linear (Manual)", "R² = 0.30", "R (manual)")
        with col_b:
            st.metric("Bayesian (brms)", "R² = 0.17", "R (brms package)")
        with col_c:
            st.metric("MLP Neural Network", "R² = 0.69", "Python (manual)")

        st.markdown("""
        ## Key Finding
        Linear Bayesian models are interpretable but hit a ceiling due to the nonlinear
        relationship between voice features and motor severity. A manually-implemented
        MLP neural network significantly outperformed both Bayesian models,
        achieving an R² of **0.69** — demonstrating the power of deep nonlinear modeling
        for this task.
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

        [📥 Dataset Link](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
        """)

        st.markdown("## Features Used")
        features = ["Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE",
                    "age", "sex", "test_time"]
        for f in features:
            st.markdown(f"- `{f}`")

        st.markdown("## Methods")
        st.markdown("""
        - Bayesian Linear Regression (manual, R)
        - Bayesian Linear Regression (brms, R)
        - MLP Neural Network (manual, Python)
        """)

# ============================================================
# PAGE 2: INTERACTIVE VISUALIZATIONS
# ============================================================
elif page == "📊 Interactive Visualizations":
    st.title("📊 Interactive Data Visualizations")
    st.markdown("Explore the Parkinson's dataset and model results interactively.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Feature Explorer", "Patient Progression", "Model Comparison"])

    # --- TAB 1: Feature vs Target ---
    with tab1:
        st.subheader("Voice Feature vs Motor UPDRS")
        st.markdown("Select a voice feature to see its relationship with motor severity.")

        feature_cols = [c for c in data.columns if c not in
                        ["motor_UPDRS", "total_UPDRS", "subject#", "index"]]

        selected_feature = st.selectbox("Choose a feature:", feature_cols, index=feature_cols.index("PPE") if "PPE" in feature_cols else 0)

        color_by = st.radio("Color points by:", ["motor_UPDRS", "age", "sex"], horizontal=True)

        fig = px.scatter(
            data,
            x=selected_feature,
            y="motor_UPDRS",
            color=color_by,
            opacity=0.4,
            trendline="ols",
            color_continuous_scale="Viridis",
            labels={"motor_UPDRS": "Motor UPDRS Score"},
            title=f"{selected_feature} vs Motor UPDRS"
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

        corr = data[selected_feature].corr(data["motor_UPDRS"])
        st.info(f"📈 Pearson correlation between **{selected_feature}** and **motor_UPDRS**: `{corr:.4f}`")

    # --- TAB 2: Patient Progression ---
    with tab2:
        st.subheader("Individual Patient Progression Over Time")
        st.markdown("Track how a patient's motor UPDRS score changes across recording sessions.")

        subject_ids = sorted(data["subject#"].unique())
        selected_subject = st.selectbox("Select a patient (subject #):", subject_ids)

        patient_df = data[data["subject#"] == selected_subject].sort_values("test_time")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=patient_df["test_time"],
            y=patient_df["motor_UPDRS"],
            mode="lines+markers",
            name="motor UPDRS",
            line=dict(color="royalblue", width=2),
            marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=patient_df["test_time"],
            y=patient_df["total_UPDRS"],
            mode="lines+markers",
            name="total UPDRS",
            line=dict(color="tomato", width=2, dash="dash"),
            marker=dict(size=6)
        ))
        fig2.update_layout(
            title=f"Patient {selected_subject}: UPDRS Over Time",
            xaxis_title="Test Time (days)",
            yaxis_title="UPDRS Score",
            legend=dict(orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Recordings", len(patient_df))
        col2.metric("Avg Motor UPDRS", f"{patient_df['motor_UPDRS'].mean():.1f}")
        col3.metric("Age", int(patient_df['age'].iloc[0]))

    # --- TAB 3: Model Comparison ---
    with tab3:
        st.subheader("Model Performance Comparison")

        models = ["Bayesian Linear\n(brms)", "Bayesian Linear\n(Manual)", "MLP Neural\nNetwork"]
        r2_scores = [0.166, 0.30, 0.69]
        rmse_scores = [7.21, 6.85, 4.32]  # approximate
        colors = ["#636EFA", "#EF553B", "#00CC96"]

        col1, col2 = st.columns(2)

        with col1:
            fig3 = go.Figure(go.Bar(
                x=models,
                y=r2_scores,
                marker_color=colors,
                text=[f"{v:.2f}" for v in r2_scores],
                textposition="outside"
            ))
            fig3.update_layout(
                title="R² Score by Model (higher = better)",
                yaxis=dict(range=[0, 1], title="R²"),
                xaxis_title="Model"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = go.Figure(go.Bar(
                x=models,
                y=rmse_scores,
                marker_color=colors,
                text=[f"{v:.2f}" for v in rmse_scores],
                textposition="outside"
            ))
            fig4.update_layout(
                title="RMSE by Model (lower = better)",
                yaxis_title="RMSE",
                xaxis_title="Model"
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""
        **Takeaway:** The MLP dramatically outperforms both linear Bayesian models,
        highlighting that the relationship between voice features and motor UPDRS is
        fundamentally nonlinear. However, the Bayesian models offer greater
        interpretability through posterior distributions on coefficients.
        """)

# ============================================================
# PAGE 3: MLP DEMO
# ============================================================
elif page == "🤖 MLP Demo":
    st.title("🤖 Live MLP Prediction Demo")
    st.markdown("Adjust voice feature values below and get a predicted motor UPDRS score from our trained MLP.")
    st.divider()

    # Train a quick model on load
    @st.cache_resource
    def train_model():
        np.random.seed(42)
        drop_cols = ["motor_UPDRS", "total_UPDRS", "subject#", "index"]
        drop_cols = [c for c in drop_cols if c in data.columns]
        X = data.drop(columns=drop_cols).values
        y = data["motor_UPDRS"].values.reshape(-1, 1)

        n = X.shape[0]
        indices = np.random.permutation(n)
        train_size = int(0.8 * n)
        train_idx = indices[:train_size]

        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]

        X_mean = X_train_raw.mean(axis=0)
        X_std = X_train_raw.std(axis=0)
        X_std[X_std == 0] = 1

        X_train = (X_train_raw - X_mean) / X_std
        y_mean = y_train_raw.mean()
        y_std = y_train_raw.std()
        y_train = (y_train_raw - y_mean) / y_std

        d = X_train.shape[1]
        h1, h2 = 64, 32
        eta = 0.01
        epochs = 500  # fewer for speed in demo
        batch_size = 64

        W1 = np.random.randn(d, h1) * np.sqrt(2 / d)
        b1 = np.zeros((1, h1))
        W2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
        b2 = np.zeros((1, h2))
        W3 = np.random.randn(h2, 1) * np.sqrt(2 / h2)
        b3 = np.zeros((1, 1))

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                Xb = X_train[perm[i:i+batch_size]]
                yb = y_train[perm[i:i+batch_size]]

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

        feature_names = [c for c in data.columns if c not in ["motor_UPDRS", "total_UPDRS", "subject#", "index"]]
        return (W1, b1, W2, b2, W3, b3), X_mean, X_std, y_mean, y_std, feature_names

    with st.spinner("Training MLP model... (one-time, ~10 seconds)"):
        weights, X_mean, X_std, y_mean, y_std, feature_names = train_model()

    W1, b1, W2, b2, W3, b3 = weights

    def predict_one(x_raw):
        x = (x_raw - X_mean) / X_std
        H1 = np.maximum(0, x @ W1 + b1)
        H2 = np.tanh(H1 @ W2 + b2)
        y_scaled = H2 @ W3 + b3
        return float(y_scaled * y_std + y_mean)

    st.markdown("### Adjust Feature Values")
    st.markdown("Use the sliders to set feature values and see the predicted motor UPDRS score.")

    # Use dataset medians as defaults
    defaults = data[feature_names].median()

    col1, col2 = st.columns(2)
    user_inputs = {}

    key_features = ["age", "sex", "test_time", "Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE"]
    key_features = [f for f in key_features if f in feature_names]
    other_features = [f for f in feature_names if f not in key_features]

    for i, feat in enumerate(key_features):
        col = col1 if i % 2 == 0 else col2
        mn = float(data[feat].min())
        mx = float(data[feat].max())
        default = float(defaults[feat])
        user_inputs[feat] = col.slider(feat, min_value=mn, max_value=mx, value=default, key=feat)

    # Fill remaining features with medians
    for feat in other_features:
        user_inputs[feat] = float(defaults[feat])

    x_input = np.array([user_inputs[f] for f in feature_names]).reshape(1, -1)
    prediction = predict_one(x_input)
    prediction = max(0, min(108, prediction))

    st.divider()
    st.markdown("### Prediction")
    col_pred, col_gauge = st.columns([1, 2])

    with col_pred:
        st.metric("Predicted Motor UPDRS", f"{prediction:.1f}", help="Scale: 0 (no symptoms) to 108 (severe)")
        severity = "Mild" if prediction < 20 else "Moderate" if prediction < 35 else "Severe"
        color = "green" if prediction < 20 else "orange" if prediction < 35 else "red"
        st.markdown(f"**Severity:** :{color}[{severity}]")

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Motor UPDRS Score"},
            gauge={
                "axis": {"range": [0, 108]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 20], "color": "lightgreen"},
                    {"range": [20, 35], "color": "orange"},
                    {"range": [35, 108], "color": "salmon"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": prediction}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
