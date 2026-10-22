import os
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = "fraud_model.pkl"
DATA_PATH = "transactions.csv"
FEATURES = ["V1", "V2", "V3", "Amount"]

# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f172a, #1d4ed8);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem;}
    .hero p {margin: .5rem 0 0 0; color: #e2e8f0; font-size: 1rem;}
    .small-note {color: #64748b; font-size: .9rem;}
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px; min-height: 2.8rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
      <h1>💳 Credit Card Fraud Detection</h1>
      <p>Machine-learning demo • Single transaction checks • CSV batch analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Use this educational app to explore predictions from your trained Random Forest model. "
    "It is a demonstration, not a real banking fraud-detection service."
)

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Keep fraud_model.pkl in the same folder as app.py.")
    st.info("Run train_model.py first to create the model file.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load fraud_model.pkl: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 Project Menu")
page = st.sidebar.radio(
    "Choose a section",
    ["Predict a Transaction", "Upload Multiple Transactions", "Dataset Overview"]
)
st.sidebar.divider()
st.sidebar.caption("Model: Random Forest")
st.sidebar.caption("Input columns: V1, V2, V3, Amount")

# ---------------- DATASET SUMMARY ----------------
def load_dataset():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None

dataset = load_dataset()

# ---------------- SINGLE TRANSACTION ----------------
if page == "Predict a Transaction":
    st.subheader("🔎 Check One Transaction")
    st.write(
        "Your model was trained with four numerical columns. V1, V2, and V3 are "
        "anonymized dataset features, so their original real-world meanings are not provided."
    )

    with st.expander("What are V1, V2, and V3?"):
        st.markdown(
            "- **V1, V2, V3:** anonymized numerical features from the dataset.\n"
            "- **Amount:** the transaction amount.\n\n"
            "These features must match the format used to train the model. "
            "Amount alone is not enough for this model to make a prediction."
        )

    with st.form("single_transaction_form"):
        left, right = st.columns(2)
        with left:
            v1 = st.number_input("V1 (anonymized feature)", value=0.0, format="%.6f")
            v2 = st.number_input("V2 (anonymized feature)", value=0.0, format="%.6f")
        with right:
            v3 = st.number_input("V3 (anonymized feature)", value=0.0, format="%.6f")
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, format="%.2f")
        submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)

    if submitted:
        row = pd.DataFrame([[v1, v2, v3, amount]], columns=FEATURES)
        try:
            prediction = int(model.predict(row)[0])
            st.markdown("### Prediction")
            if prediction == 1:
                st.error("⚠️ The model predicts: Potentially Fraudulent")
            else:
                st.success("✅ The model predicts: Normal Transaction")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(row)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    fraud_probability = float(probabilities[classes.index(1)]) * 100
                    st.metric("Model-estimated fraud probability", f"{fraud_probability:.1f}%")
                    st.caption(
                        "This is the model's estimated probability, not a guarantee or a reliable real-world risk score."
                    )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.divider()
    st.caption(
        "Tip: use real feature values from a compatible dataset row for a meaningful demonstration. "
        "Arbitrary values only show how the app works."
    )

# ---------------- CSV BATCH PREDICTION ----------------
elif page == "Upload Multiple Transactions":
    st.subheader("📂 Predict Many Transactions")
    st.write(
        "Upload a CSV file containing these exact columns: **V1, V2, V3, Amount**. "
        "An optional **Class** column can be included for comparison with known labels."
    )

    sample = pd.DataFrame(
        [[0.1, -0.2, 0.3, 50.0], [-1.0, 0.4, -0.5, 250.0]],
        columns=FEATURES
    )
    st.download_button(
        "⬇️ Download sample CSV template",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="transaction_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a transactions CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            incoming = pd.read_csv(uploaded_file)
            missing = [column for column in FEATURES if column not in incoming.columns]

            if missing:
                st.error("Missing required column(s): " + ", ".join(missing))
                st.write("Your CSV columns are:", ", ".join(map(str, incoming.columns)))
            elif incoming.empty:
                st.warning("The uploaded CSV has no transaction rows.")
            else:
                numeric_data = incoming[FEATURES].apply(pd.to_numeric, errors="coerce")
                if numeric_data.isna().any().any():
                    st.error("Some required feature values are empty or non-numeric. Please correct the CSV and upload it again.")
                else:
                    predictions = model.predict(numeric_data)
                    output = incoming.copy()
                    output["Predicted_Class"] = predictions
                    output["Predicted_Result"] = [
                        "Potentially Fraudulent" if int(value) == 1 else "Normal"
                        for value in predictions
                    ]

                    st.success(f"Finished predicting {len(output)} transaction(s).")
                    fraud_count = int((output["Predicted_Class"] == 1).sum())
                    normal_count = len(output) - fraud_count
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Transactions checked", len(output))
                    m2.metric("Predicted normal", normal_count)
                    m3.metric("Predicted potentially fraudulent", fraud_count)

                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "⬇️ Download prediction results",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

                    if "Class" in output.columns:
                        st.info(
                            "The uploaded file includes a Class column. Predictions are in Predicted_Class; "
                            "Class contains the existing labels. These are not automatically treated as model evaluation."
                        )
        except Exception as error:
            st.error(f"Could not read this CSV file: {error}")

# ---------------- DATASET OVERVIEW ----------------
elif page == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    if dataset is None:
        st.warning(
            "transactions.csv was not found or could not be read. Place it in the same folder as app.py "
            "to see a dataset summary."
        )
    else:
        st.write("Summary of the dataset file found beside app.py.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", int(dataset.shape[0]))
        c2.metric("Columns", int(dataset.shape[1]))
        missing_values = int(dataset.isna().sum().sum())
        c3.metric("Missing values", missing_values)

        if "Class" in dataset.columns:
            counts = dataset["Class"].value_counts().sort_index()
            normal_count = int(counts.get(0, 0))
            fraud_count = int(counts.get(1, 0))
            a, b = st.columns(2)
            a.metric("Normal labels (Class = 0)", normal_count)
            b.metric("Fraud labels (Class = 1)", fraud_count)
            st.bar_chart(pd.DataFrame({
                "Transactions": {
                    "Normal (0)": normal_count,
                    "Fraud (1)": fraud_count
                }
            }))

        st.markdown("#### Preview")
        st.dataframe(dataset.head(10), use_container_width=True)

        st.markdown("#### Data quality")
        st.dataframe(
            pd.DataFrame({
                "Column": dataset.columns,
                "Data type": [str(dataset[column].dtype) for column in dataset.columns],
                "Missing values": [int(dataset[column].isna().sum()) for column in dataset.columns]
            }),
            use_container_width=True
        )

    st.markdown("#### Model performance reminder")
    st.warning(
        "The earlier test on this project missed the fraud example in the test split. "
        "The dataset is very small (100 rows with only 2 fraud examples), so this model is not reliable "
        "for real-world decisions. Dataset counts shown above are not model accuracy metrics."
    )

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    '<p class="small-note">Student project demo • Random Forest • Always verify predictions independently.</p>',
    unsafe_allow_html=True
)
