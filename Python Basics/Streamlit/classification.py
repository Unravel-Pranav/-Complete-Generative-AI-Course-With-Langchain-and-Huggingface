import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import time

# Function to load data and cache it
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train the model using the first 4 columns (features)
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Title and fun intro
st.title("🌸 Iris Flower Species Predictor")
st.subheader("Predict the type of Iris flower based on its features!")
st.write("Adjust the sliders below to select the flower's dimensions, and we'll guess which species it is. 🌼")

# Sidebar sliders for input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("🌿 Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), step=0.1)
sepal_width = st.sidebar.slider("🌿 Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), step=0.1)
petal_length = st.sidebar.slider("🌸 Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), step=0.1)
petal_width = st.sidebar.slider("🌸 Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), step=0.1)

# Displaying the current feature values
st.write("### Your Iris Flower's Features")
st.write(f"**Sepal Length**: {sepal_length} cm")
st.write(f"**Sepal Width**: {sepal_width} cm")
st.write(f"**Petal Length**: {petal_length} cm")
st.write(f"**Petal Width**: {petal_width} cm")

input_data = [sepal_length, sepal_width, petal_length, petal_width]

# Button for making the prediction
if st.button("🌟 Predict the Species"):
    st.write("Making the prediction...")

    # Add a fun progress bar to simulate prediction time
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.001)  # Adding a slight delay to make it fun
        progress_bar.progress(percent + 1)

    # Prediction
    prediction = model.predict([input_data])
    prediction_species = target_names[prediction[0]]

    # Display prediction result with some fun text
    st.success(f"🎉 The predicted species is **{prediction_species}**!")

    # Optionally, add an image of the predicted Iris species (you can download images and store them locally)
    if prediction_species == 'setosa':
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg", caption="Iris Setosa", width=300)
    elif prediction_species == 'versicolor':
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor", width=300)
    elif prediction_species == 'virginica':
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica", width=300)

st.sidebar.markdown("---")
st.sidebar.write("🔍 **Explore more features below!**")
st.sidebar.write("👈 Adjust the sliders to see live updates.")

# Sidebar fun facts
st.sidebar.markdown("### 🌱 Did you know?")
st.sidebar.write("- **Iris Setosa** is often found in wetlands.")
st.sidebar.write("- **Iris Versicolor** has bluish-purple petals.")
st.sidebar.write("- **Iris Virginica** is native to the eastern United States.")
