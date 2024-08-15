import streamlit as st
import numpy as np
import pickle
import pandas as pd

model_path = "ML_model.sav"
loaded_model = pickle.load(open(model_path, "rb"))


def generate_synthetic_data(num_samples):
    np.random.seed(0)
    data = {
        'Age': np.random.uniform(20, 70, num_samples),
        'Sleep_Hours': np.random.uniform(4, 10, num_samples),
        'Physical_Activity_Level': np.random.uniform(1, 10, num_samples),
        'Heart_Rate': np.random.uniform(60, 100, num_samples),
        'Daily_Steps': np.random.uniform(1000, 15000, num_samples)
    }
    df = pd.DataFrame(data)
    return df


def predict_stress_level(input_data):
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]


def main():
    st.title("Stress Level Prediction Web App")

    st.header("Generate Synthetic Data and Predict Stress Levels")
    num_samples = st.number_input("Enter the number of synthetic samples to generate", min_value=1, max_value=1000,
                                  value=100)

    if st.button("Generate and Predict"):
        data = generate_synthetic_data(num_samples)
        st.write("Synthetic Data Sample:")
        st.write(data.head())

        data['Predicted_Stress_Level'] = data.apply(lambda row: predict_stress_level(row), axis=1)
        st.write("Predictions for Synthetic Data:")
        st.write(data)

        mean_prediction = data['Predicted_Stress_Level'].mean()
        st.write(f"Mean Predicted Stress Level for Synthetic Data: {mean_prediction:.2f}")

    st.header("Manual Input")
    Age = st.text_input("Enter Age")
    Sleep_Hours = st.text_input("Enter Sleep Hours")
    Physical_Activity_Level = st.text_input("Enter Physical Activity Level")
    Heart_Rate = st.text_input("Enter Heart Rate")
    Daily_Steps = st.text_input("Enter Daily Steps")

    if st.button("Predict Stress Level"):
        try:
            Age = float(Age)
            Sleep_Hours = float(Sleep_Hours)
            Physical_Activity_Level = float(Physical_Activity_Level)
            Heart_Rate = float(Heart_Rate)
            Daily_Steps = float(Daily_Steps)

            diagnosis = predict_stress_level([Age, Sleep_Hours, Physical_Activity_Level, Heart_Rate, Daily_Steps])
            st.success(f"Predicted Stress Level: {diagnosis}")

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")


if __name__ == "__main__":
    main()
