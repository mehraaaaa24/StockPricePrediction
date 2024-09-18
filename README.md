**Stock Market Predictor**
This project leverages machine learning to predict stock prices using LSTM (Long Short-Term Memory) neural networks. The model is trained on historical stock data and visualized using various technical indicators such as moving averages (MA50, MA100, MA200). The application is deployed as a real-time web app using Streamlit, allowing users to input a stock symbol and view the model's predictions alongside actual stock prices.

Tech Stack:
Python: Core programming language
TensorFlow & Keras: For building and training the neural network model
Pandas: For data manipulation and preprocessing
NumPy: For numerical operations
yfinance: To fetch historical stock data
Matplotlib: For visualizing stock trends and predictions
MinMaxScaler (from sklearn): For data normalization
Streamlit: For creating a real-time web application interface
Jupyter Notebook: For development and experimentation
Features:
Predict stock prices using a pre-trained LSTM model
Compare real prices with predictions
Visualize stock prices alongside moving averages (MA50, MA100, MA200)
Real-time data input via Streamlit web interface
