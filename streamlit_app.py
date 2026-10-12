from pathlib import Path
import pickle

import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "data.pkl"


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(
    page_title="Value predictor",
    page_icon=":material/functions:",
)

st.title("Value predictor")
st.caption("Estimate a result from one input using the saved regression model.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The model file data.pkl is missing from the app folder.")
    st.stop()
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

with st.form("prediction"):
    value = st.slider(
        "Input value",
        min_value=1.0,
        max_value=200.0,
        step=0.2,
        format="%0.1f",
        help="The model uses a single feature.",
    )
    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        icon=":material/calculate:",
    )

if submitted:
    prediction = float(model.predict([[value]])[0])
    st.metric("Predicted value", f"{prediction:.4f}")
