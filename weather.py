import streamlit as st
import pandas as pd

@st.cache
def load_data():
    # Load the data into a pandas dataframe
    return pd.read_csv("city_pollution.csv")

def main():
    st.title("Air Pollution Calculator")
    data = load_data()
    st.write("Data shape: ", data.shape)

    # Show data table
    st.dataframe(data)

    # Filter data by city
    city = st.selectbox("Select a city:", data["City"].unique())
    city_data = data[data["City"] == city]

    # Show highest pollution value
    highest_pollution = city_data["Pollution"].max()
    st.write("Highest pollution in", city, "is", highest_pollution)

    #
    Data = [1,2,3,4]
    st.write(Data)

if __name__ == '__main__':
    main()
