from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AD ML App", layout="wide")
st.title("🔬 Anaerobic Digestion ML App")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- INIT SESSION ----------------
if "trained" not in st.session_state:
    st.session_state["trained"] = False

# ---------------- LOAD DATA ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.sidebar.selectbox("Select Target", df.columns)

    selected_models = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "SVR", "Linear Regression", "KNN", "XGBoost", "ANN"],
        default=["Random Forest", "XGBoost"]
    )

    train_button = st.sidebar.button("🚀 Train Models")

    # ---------------- TRAIN ----------------
    if train_button:

        if len(selected_models) == 0:
            st.warning("Please select at least one model")
            st.stop()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        feature_names = X.columns

        all_models = {
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "ANN": MLPRegressor(max_iter=500)
        }

        models = {name: all_models[name] for name in selected_models}

        results = []
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append([name, r2, rmse])
            trained_models[name] = model

        results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE"])

        # Best model
        best_row = results_df.loc[results_df["R2 Score"].idxmax()]
        best_model_name = best_row["Model"]
        best_model = trained_models[best_model_name]
        best_pred = best_model.predict(X_test_scaled)

        # Predictions dataframe
        # Convert X_test_scaled back to original values using index
        X_test_original = X.iloc[y_test.index].reset_index(drop=True)

        pred_df = X_test_original.copy()

        pred_df["Actual"] = y_test.reset_index(drop=True)
        pred_df["Best_Model_Predicted"] = best_pred

        # Save everything in session
        st.session_state.update({
            "trained": True,
            "X": X,
            "scaler": scaler,
            "results_df": results_df,
            "trained_models": trained_models,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_pred": best_pred,
            "y_test": y_test,
            "pred_df": pred_df
        })

# ---------------- AFTER TRAIN ----------------
if st.session_state["trained"]:

    X = st.session_state["X"]
    scaler = st.session_state["scaler"]
    results_df = st.session_state["results_df"]
    trained_models = st.session_state["trained_models"]
    best_model = st.session_state["best_model"]
    best_model_name = st.session_state["best_model_name"]
    best_pred = st.session_state["best_pred"]
    y_test = st.session_state["y_test"]
    pred_df = st.session_state["pred_df"]

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance",
        "📈 Predictions",
        "🔥 Feature Importance",
        "🧠 SHAP",
        "🔮 Predict New Data"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Model Performance")

        def highlight_best(row):
            return ['background-color: lightgreen'] * len(row) if row["Model"] == best_model_name else [''] * len(row)

        st.dataframe(results_df.style.apply(highlight_best, axis=1))

        st.success(f"🏆 Best Model: {best_model_name}")

        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["R2 Score"])
        plt.xticks(rotation=30)
        st.pyplot(fig)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader(f"Actual vs Predicted ({best_model_name})")

        fig, ax = plt.subplots()
        ax.scatter(y_test, best_pred)

        min_val = min(min(y_test), min(best_pred))
        max_val = max(max(y_test), max(best_pred))

        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color = 'red')

        st.pyplot(fig)

        st.dataframe(pred_df.head())

        csv = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    # ---------------- TAB 3 ----------------
    with tab3:

        if "Random Forest" in trained_models:
            rf_model = trained_models["Random Forest"]

            feat_imp = pd.DataFrame({
                "Feature": X.columns,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots()
            ax.barh(feat_imp["Feature"], feat_imp["Importance"])
            st.pyplot(fig)

    # ---------------- TAB 4 ----------------
    with tab4:

        if "Random Forest" in trained_models:
            rf_model = trained_models["Random Forest"]

            explainer = shap.TreeExplainer(rf_model)
            X_sample = scaler.transform(X.iloc[:50])

            shap_values = explainer.shap_values(X_sample)

            fig = plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)
            st.pyplot(fig)

        # ---------------- TAB 5 ----------------
    with tab5:

        st.subheader("🔮 Enter Input Values")

        with st.form("prediction_form"):

            input_data = {}

            for col in X.columns:
                input_data[col] = st.text_input(f"{col}", "")

            submit = st.form_submit_button("Predict")

            if submit:

                input_df = pd.DataFrame([input_data])

                # Handle missing + type conversion
                for col in input_df.columns:
                    if input_df[col].iloc[0] == "":
                        input_df[col] = X[col].mean()
                    else:
                        input_df[col] = float(input_df[col])

                input_scaled = scaler.transform(input_df)

                prediction = best_model.predict(input_scaled)

                st.success(f"🎯 Predicted Value: {prediction[0]:.4f}")

                # ---------------- SHOW INPUT ----------------
                st.subheader("🧾 Entered Input Data")
                st.dataframe(input_df)

                # ---------------- SAVE HISTORY ----------------
                result_row = input_df.copy()
                result_row["Predicted_Output"] = prediction[0]

                if "prediction_history" not in st.session_state:
                    st.session_state["prediction_history"] = result_row
                else:
                    st.session_state["prediction_history"] = pd.concat(
                        [st.session_state["prediction_history"], result_row],
                        ignore_index=True
                    )

        # ---------------- DISPLAY HISTORY ----------------
        if "prediction_history" in st.session_state:

            st.subheader("📊 Prediction History")
            st.dataframe(st.session_state["prediction_history"])

            # ---------------- DOWNLOAD HISTORY ----------------
            csv_hist = st.session_state["prediction_history"].to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Prediction History",
                csv_hist,
                "prediction_history.csv",
                "text/csv"
            )