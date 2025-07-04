# Stock-Price-Prediction-Streamlit-App

## Overview
The stock market is one of the most dynamic and complex fields of finance, where the stock prices of companies continuously fluctuate. This project aims to develop a stock price prediction system using linear regression for top companies. The system is implemented as a web application using Streamlit, which provides an intuitive and interactive interface for users to analyze stock prices and make informed decisions. The project consists of two main components: stock price prediction and stock analysis using technical indicators. For stock price prediction, a linear regression model is trained using historical stock data, taking into account various factors such as company financials, market trends, and historical prices. The trained model is then deployed using Streamlit, allowing users to input relevant information and obtain predictions for future stock prices.

## Key Features
* **Visualize Historical Data:** Users can visualize historical stock price data using interactive charts, exploring trends and fluctuations over a specified time period.
* **Technical Indicator Analysis:** Incorporates various technical indicators, including Bollinger Bands, MACD, RSI, SMA, and EMA, to provide insights into stock price movement and market trends.
* **Recent Data Display:** Provides quick access to the most recent data of the selected stock, offering an up-to-date snapshot of its performance.
* **Stock Price Prediction:** Includes a prediction feature that utilizes linear regression for forecasting future stock prices based on historical data.
* **User-Friendly Interface:** Offers a clear navigation and intuitive controls, implemented with Streamlit, for a seamless user experience.
* **Customizable Stock Selection:** Allows users to select and analyze stocks from a predefined list of top companies (e.g., Google, Microsoft, Tesla, Airbnb, Meta).
* **Performance Evaluation Metrics:** Displays evaluation metrics such as R-squared score and Mean Absolute Error to assess the prediction model's accuracy and reliability.
* **Responsive Design:** Designed to be responsive, ensuring smooth user interactions and fast loading times across different devices.

## Screenshots
### Main Interface
![Main App Interface](images/app_screenshot_main.png)

### Prediction Output
![Prediction Results](images/app_screenshot_prediction.png)

### Technical Indicators
![Technical Indicators Chart](images/app_screenshot_indicators.png)

### User-Case Diagram
![User-Case Diagram](images/user_case_diagram.png)

### Architecture Diagram
![Architecture Diagram](images/architecture_diagram.png)

### Dataset Screenshot
![Dataset Screenshot](images/dataset_screenshot.png)

## Technologies Used
* Python
* Streamlit (Web Application Framework)
* Pandas (Data Manipulation)
* yfinance (Stock Data Retrieval)
* ta (Technical Analysis Library)
* Scikit-learn (Machine Learning)

## How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zeeshan510/Stock-Price-Prediction-Streamlit-App.git
    cd Stock-Price-Prediction-Streamlit-App
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You will create `requirements.txt` later, after you've put your `stock_app.py` in the repo.)
3.  **Run the Streamlit app:**
    ```bash
    streamlit run stock_app.py
    ```
    The application will open in your default web browser.

## Project Documentation
* [Full Project Report (PDF)](docs/Stock_Price_Prediction_Report.pdf)
* [Project Presentation (PDF)](docs/Stock_Price_Prediction_Presentation.pdf)

## Developed By
* Mohammed Abdul Zeeshan

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
