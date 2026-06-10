import streamlit as st
import pandas as pd
import numpy as np

st.title("Test Streamlit")

st.header("Graphique")
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["A", "B", "C"]
)
st.line_chart(data)

st.header("Widgets")
name = st.text_input("Ton nom")
if name:
    st.success(f"Bonjour, {name} !")

slider_val = st.slider("Valeur", 0, 100, 50)
st.write(f"Valeur sélectionnée : {slider_val}")

st.header("Tableau")
st.dataframe(data.head())
