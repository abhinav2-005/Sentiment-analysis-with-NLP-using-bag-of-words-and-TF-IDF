import streamlit as st
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Emotion Classification",
    page_icon="😊",
    layout="centered"
)


# -----------------------------------
# Load Model and Vectorizer
# -----------------------------------
@st.cache_resource
def load_files():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")

    return vectorizer, model


vectorizer, model = load_files()


# -----------------------------------
# Emotion Mapping
# -----------------------------------
emotion_mapping = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}


emotion_emoji = {
    "Sadness": "😢",
    "Anger": "😡",
    "Love": "❤️",
    "Surprise": "😲",
    "Fear": "😨",
    "Joy": "😊"
}


# -----------------------------------
# Header
# -----------------------------------
st.title("Emotion Classification")

st.write(
    "Enter a sentence and the trained machine learning "
    "model will predict the emotion expressed in the text."
)

st.divider()


# -----------------------------------
# Text Input
# -----------------------------------
st.subheader("Enter Text")

text = st.text_area(
    "Text",
    placeholder="Example: I am so happy to see my friends today!",
    height=180
)


# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Predict Emotion", type="primary"):

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        # Convert text into TF-IDF features
        text_vectorized = vectorizer.transform([text])

        # Get prediction probabilities
        probabilities = model.predict_proba(
            text_vectorized
        )[0]

        # Find class with highest probability
        max_index = probabilities.argmax()

        predicted_class = model.classes_[max_index]

        predicted_emotion = emotion_mapping.get(
            int(predicted_class),
            str(predicted_class)
        )

        max_probability = probabilities[max_index] * 100

        emoji = emotion_emoji.get(
            predicted_emotion,
            "🤖"
        )

        # -----------------------------------
        # Display Result
        # -----------------------------------
        st.subheader("Predicted Emotion")

        st.success(
            f"{emoji} **{predicted_emotion}**"
        )

# -----------------------------------
# Footer
# -----------------------------------
st.divider()

st.caption(
    "Built with Python, Scikit-learn, TF-IDF, "
    "Logistic Regression and Streamlit"
)
