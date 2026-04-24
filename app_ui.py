import streamlit as st
import requests

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

# Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 5, 0)
parch = st.number_input("Parents/Children aboard", 0, 5, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

family_size = sibsp + parch

# Button
if st.button("Predict"):
    data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "family_size": family_size
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = response.json()

        if "prediction" in result:
            if result["prediction"] == 1:
                st.success("✅ Survived")
            else:
                st.error("❌ Did Not Survive")
        else:
            st.error(result)

    except:
        st.error("⚠️ API not running. Start FastAPI first.")