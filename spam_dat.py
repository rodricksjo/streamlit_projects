import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache
def load_data():
    # Load the data into a pandas dataframe
    return pd.read_csv("sms_spam.csv")

@st.cache
def train_model(data):
    # Split the data into training and test sets
    train = data.sample(frac=0.8, random_state=0)
    test = data.drop(train.index)

    # Prepare the training data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    y_train = train["spam"]

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Prepare the test data
    X_test = vectorizer.transform(test["text"])
    y_test = test["spam"]

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    return model, vectorizer, accuracy

def main():
    st.title("SMS Spam Predictor")
    data = load_data()
    st.write("Data shape: ", data.shape)

    # Show data table
    st.dataframe(data.head())

    # Train the model
    model, vectorizer, accuracy = train_model(data)
    st.write("Model accuracy: ", accuracy)

    # Get user input
    message = st.text_input("Enter a message:")
    
    print("Test")
    # Predict spam
    if message:
        message = [message]
        message = vectorizer.transform(message)
        prediction = model.predict(message)[0]
        probability = model.predict_proba(message).tolist()[0][1]
        if prediction == 1:
            st.write("This message is spam with a probability of", probability)
        else:
            st.write("This message is not spam with a probability of", probability)

if __name__ == '__main__':
    main()    