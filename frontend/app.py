import streamlit as st
import requests

st.title("🧬 Cancer Drug Repurposing GNN")
st.markdown("**MLOps Pipeline Demo**")

drug_id = st.text_input("Drug ID (e.g. DRUG1)", "DRUG1")
target_disease = st.text_input("Target Cancer Type", "CANCER_LUNG")

if st.button("Run Repurposing Prediction"):
    with st.spinner("Running GNN inference..."):
        response = requests.post(
            "http://localhost:8000/predict",
            json={"drug_id": drug_id, "target_disease": target_disease}
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Repurposing Score: **{result['repurposing_score']:.2f}**")
            st.info(result["explanation"])
        else:
            st.error("API error")