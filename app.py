import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
@st.cache_data
def load_data():
  """Load data from real housing dataset"""
  url = "USA_housing_dataset.csv" # file inside GitHub repo
  df = pd.read_csv(url)
  df['size']=df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
  return df

def preprocess_data(df):
  """Select relevant columns and clean data"""
  df = df[['size', 'bedrooms', 'price']].dropna()
  return df

def train_model(df, features):
  """Train a linear regression model with selected features"""
  X=df[features]
  y=df['price']
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

  model = LinearRegression()
  model.fit(X_train,y_train)

  return model

def main():
  st.title("House Price Prediction App (based on Kaggle's USA housing dataset)")

  st.write("Put in house details to predict its price")

  # load and preprocess data
  df=load_data()
  df=preprocess_data(df)

  # user input fields
  size = st.number_input(
    'House size',
    min_value=float(df['size'].min()),
    max_value=float(df['size'].max()),
    value=float(df['size'].median()),
    step=100
  )

  bedrooms = st.number_input(
    'Number of bedrooms',
    min_value=float(df['bedrooms'].min()),
    max_value=float(df['bedrooms'].max()),
    value=float(df['bedrooms'].median()),
    step=0.5
  )

  col1, col2 =st.columns(2)

  model1 = train_model(df,['size'])
  model2 = train_model(df,['size','bedrooms'])

  # initialize values
  fig = None

  # prediction with size only
  with col1:
    if st.button('Using size only'):
      predicted_price = model1.predict([[size]])[0]
      st.success(f'Predicted price based on size: ${predicted_price:.2f}')

    # 2d scatter plot
      fig = px.scatter(
        df,
        x='size',
        y='price',
        title='Size vs house price'
      )

    # add prediction marker
      fig.add_scatter(
        x=[size],
        y=[predicted_price],
        mode='markers',
        marker=dict(color='red',size=10),
        name='Predicted Price'
      )


  # prediction with size and bedrooms
  with col2:
    if st.button('Using size and bedrooms'):
      predicted_price = model2.predict([[size,bedrooms]])[0]
      st.success(f'Predicted price based on size and bedrooms: ${predicted_price:,.2f}')

      # 3d scatter plot
      fig = px.scatter_3d(
        df,
        x='size',
        y='bedrooms',
        z='price',
        title='House Prices in 3D',
        labels={'size': 'Size (sqft)', 'bedrooms': 'Bedrooms', 'price': 'Price'},
        opacity=0.7
      )

      # Add predicted point in red
      fig.add_trace(
        px.scatter_3d(
          pd.DataFrame({'size': [size], 'bedrooms': [bedrooms], 'price': [predicted_price]}),
          x='size',
          y='bedrooms',
          z='price'
        ).data[0].update(marker=dict(color='red', size=10)))

  # display graph (only if fig is created)
  if fig:
    st.plotly_chart(fig)


if __name__ == '__main__':
  main()