import streamlit as st
import requests

st.set_page_config(page_title="AI Quantum Solver", layout="centered")

st.title("AI Quantum Solver")
st.write("Predict the ground-state energy E0 of a disordered XXZ chain.")

st.subheader("Model parameters")
Jxy = st.number_input("Jxy", value=1.0)
Jz = st.number_input("Jz", value=1.2)
W = st.number_input("W", value=3.0)

st.subheader("Disorder fields h_i")
default_h = "0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.4, -0.9, 0.2, 0.6, -0.5, 1.0"
h_text = st.text_area("Enter h values separated by commas", value=default_h, height=100)

if st.button("Predict E0"):
    try:
        h = [float(x.strip()) for x in h_text.split(",") if x.strip() != ""]

        payload = {
            "Jxy": Jxy,
            "Jz": Jz,
            "W": W,
            "h": h
        }

        response = requests.post("http://api:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction successful")
            st.write(f"**Predicted E0:** {result['predicted_E0']:.6f}")
            st.write(f"**System size N:** {result['N']}")
        else:
            st.error(f"API error: {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Error: {e}")