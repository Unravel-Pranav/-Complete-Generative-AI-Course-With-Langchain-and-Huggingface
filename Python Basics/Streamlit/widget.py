import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns

st.title("Welcome to Streamlit")

name = st.text_input("Enter your name")
if name:
    st.write(f"Hello {name}!")
    
    
age=st.slider("How old are you?", min_value=0, max_value=100)
st.write("I'm", age, "years old")


options=["java","Python","R","C++"]
choice=st.selectbox("what's your favourite language?",options)
st.write(f"your favourite language is {choice}")
    
uploaded_file=st.file_uploader("Choose a file")
savefile=st.button("save")
if savefile:
    st.write(uploaded_file)

uploaded_file1=st.file_uploader("choose a CSV file",type="csv")

if uploaded_file1:
    df=pd.read_csv(uploaded_file1)
    st.write(df)
    
