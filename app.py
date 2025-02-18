import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
def load_data():
  """Load data from real housing dataset""""
    url = "USA_housing_dataset.csv"  # file inside GitHub repo
    df = pd.read_csv(url)
    df['size']=df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
    return df

def preprocess_data(df):
    """Select relevant columns and clean data"""
    df = df[['size', 'price']].dropna()  # Keep only size & price columns
    return df

def train_model(df):
  """Train a linear regression model"""
  X=df[['size']]
  y=df['price']
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

  model = LinearRegression()
  model.fit(X_train,y_train)

  return model

def main():
  st.title('House Price Prediction App')

  st.write("Put in your house size to predict its price (based on USA housing dataset)")

  #load and preprocess data
  df=load_data()
  df=preprocess_data(df)
  model = train_model(df)

  size = st.number_input('House size', min_value=500, max_value=5000, value=1500)

  if st.button('Predict price'):
    predicted_price = model.predict([[size]])
    st.success(f'Predicted price: ${predicted_price[0]:.2f}')


    fig = px.scatter(df,x='size',y='price',title='Size vs house price')
    fig.add_scatter(
      x=[size],
      y=[predicted_price[0]],
      mode='markers',
      marker=dict(color='red',size=15),
      name='Predicted Price'
    )

    st.plotly_chart(fig)

if __name__ == '__main__':
  main()