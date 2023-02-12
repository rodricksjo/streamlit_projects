import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache
def load_data():
    # Load the data into a pandas dataframe
    return pd.read_csv("credit_card_dat.csv")

@st.cache
def train_model(data):
    # Split the data into training and test sets
    train = data.sample(frac=0.8, random_state=0)
    test = data.drop(train.index)

    # Prepare the training data
    X_train = train.drop("Class", axis=1)
    y_train = train["Class"]

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Prepare the test data
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]

    # Evaluate the model
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

def main():
    st.title("Credit Card Fraud Predictor")
    data = load_data()
    st.write("Data shape: ", data.shape)

    # Show data table
    st.dataframe(data.head())

    # Train the model
    model, accuracy = train_model(data)
    st.write("Model accuracy: ", accuracy)

    # Get user input
    features = st.multiselect("Select features:", data.columns.tolist()[:-1])
    values = [st.number_input(f"Enter {feature}:") for feature in features]

    # Predict fraud
    if all(values):
        input_data = {feature: [value] for feature, value in zip(features, values)}
        input_data = pd.DataFrame(input_data)
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.write("This transaction is fraudulent.")
        else:
            st.write("This transaction is not fraudulent.")

if __name__ == '__main__':
    main()
