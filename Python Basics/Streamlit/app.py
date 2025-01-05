import streamlit as st
import pandas as pd
import numpy as np

#Title of the application

st.title("Welcome to Streamlit")

#Display a simple text

st.text("This is a simple text")
st.write("Hi my name is Pranav")

#Display a dataframe

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.dataframe(df)

#Create a line Chart

chart_data = pd.DataFrame(
    
    np.random.randn(20, 3),columns=['a', 'b', 'c']
)

st.line_chart(chart_data)
