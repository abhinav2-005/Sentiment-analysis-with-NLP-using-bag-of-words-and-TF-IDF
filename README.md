# 🤖 Text Classification using Machine Learning

A machine learning text classification project that uses **TF-IDF Vectorization** and **Logistic Regression** to classify text into predefined categories.

The project also includes a **Streamlit web interface** where users can enter text and get a prediction from the trained model.

## 🚀 Features

* Text classification using Machine Learning
* TF-IDF based text feature extraction
* Logistic Regression classification model
* Prediction probability display
* Interactive Streamlit frontend
* Pre-trained model and vectorizer stored using Joblib

## 🧠 Machine Learning Pipeline

The application follows this workflow:

```text
User Input
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression Model
    ↓
Prediction
    ↓
Prediction Probability
```

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Streamlit

## 📁 Project Structure

```text
Text-Classification/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
└── README.md
```

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
pip install streamlit scikit-learn joblib
```

For compatibility with the version used when the model was saved:

```bash
pip install scikit-learn==1.6.1
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Enter text into the text box and click **Predict** to generate a classification result.

## 📊 Model

The project uses:

### TF-IDF Vectorizer

The `TfidfVectorizer` converts text into numerical TF-IDF features that can be processed by the machine learning model.

### Logistic Regression

A trained `LogisticRegression` model is used to classify the vectorized text.

The saved files are:

```text
vectorizer.pkl
model.pkl
```

## 🔍 Prediction

The Streamlit application displays:

* Predicted class
* Probability for each available class

The probability values are generated using the trained Logistic Regression model.

## 🖥️ Streamlit Interface

The frontend provides:

1. Text input area
2. Prediction button
3. Predicted class
4. Class probabilities

## ⚠️ Model Compatibility

The trained `.pkl` files were created using scikit-learn 1.6.1.

Using a significantly different scikit-learn version may produce compatibility warnings or unexpected behavior. Using the same version used during training is recommended.

## 🎯 Project Purpose

This project demonstrates a complete basic NLP classification workflow:

* Converting text into numerical features
* Training a machine learning classifier
* Saving trained components
* Loading the trained components
* Making predictions on new text
* Building a user interface with Streamlit
