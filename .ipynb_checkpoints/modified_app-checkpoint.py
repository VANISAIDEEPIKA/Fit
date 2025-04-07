import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
calories_data = pd.read_csv("calories_with_burned_column.csv")
exercise_data = pd.read_csv("exercise.csv")

# Merge datasets if 'User_ID' exists
if 'User_ID' in exercise_data.columns and 'User_ID' in calories_data.columns:
    data = pd.merge(exercise_data, calories_data, on="User_ID")
else:
    st.error("Error: 'User_ID' column not found in one of the datasets.")
    st.stop()

# Fix column naming issues
rename_map = {"Age_x": "Age", "Duration_x": "Duration", "Heart_Rate_x": "Heart_Rate"}
data.rename(columns=rename_map, inplace=True, errors="ignore")
data.drop(columns=["Age_y", "Duration_y", "Heart_Rate_y"], errors="ignore", inplace=True)

# Ensure gender is numeric
data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})

# Drop missing values
data.dropna(inplace=True)

# Streamlit UI
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.write("Track your fitness progress, calculate BMI, and predict calorie burn!")

# Sidebar input for user details
st.sidebar.header("User Input Parameters")

def user_input_features():
    height = st.sidebar.number_input("Height (cm):", 100, 220, 170)
    weight = st.sidebar.number_input("Weight (kg):", 30, 150, 70)
    age = st.sidebar.slider("Age:", 10, 100, 30)
    gender = st.sidebar.radio("Gender:", ["Male", "Female"])
    duration = st.sidebar.slider("Exercise Duration (min):", 0, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm):", 40, 200, 80)

    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    st.sidebar.write(f"### Your BMI: {bmi:.2f}")

    # Convert gender to numeric
    gender_encoded = 0 if gender == "Male" else 1

    return pd.DataFrame([[age, gender_encoded, bmi, duration, heart_rate]], 
                        columns=["Age", "Gender", "BMI", "Duration", "Heart_Rate"])

user_data = user_input_features()

# Verify required columns exist
required_columns = ["Age", "BMI", "Duration", "Heart_Rate", "Gender", "Calories_Burned"]
if not all(col in data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    st.error(f"Error: Missing columns in dataset: {missing_cols}")
    st.stop()

# Model Training & Prediction
X = data[["Age", "BMI", "Duration", "Heart_Rate", "Gender"]]
y = data["Calories_Burned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(user_data)
st.subheader("🔥 Predicted Calories Burned:")
st.write(f"### {prediction[0]:.2f} kcal")

# Exercise Recommendations
st.subheader("💡 Exercise Recommendations")
bmi_value = user_data["BMI"][0]
if bmi_value < 18.5:
    st.write("🔹 Underweight: Focus on strength training and high-calorie nutrition.")
elif 18.5 <= bmi_value < 24.9:
    st.write("✅ Healthy BMI: Maintain a balanced routine with cardio and strength exercises.")
elif 25 <= bmi_value < 29.9:
    st.write("⚠️ Overweight: Prioritize cardio workouts like running, swimming, or cycling.")
else:
    st.write("🚨 Obese: Engage in high-intensity workouts and monitor your diet strictly.")

# Data Visualization
st.subheader("📊 Data Insights")
fig, ax = plt.subplots()
sns.scatterplot(x=data["Duration"], y=data["Calories_Burned"], ax=ax)
st.pyplot(fig)
