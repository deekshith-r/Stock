import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from stocktwits import StockTwits
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LinearRegression

nltk.download('vader_lexicon')

# Initialize StockTwits client and sentiment analyzer
stocktwits = StockTwits()
sia = SentimentIntensityAnalyzer()

# Download historical stock data
data = yf.download("^GSPC", start="2018-01-01", interval='1d')

# Filter required data and preprocess
data = data[['Close', 'Volume']]
data.sort_index(inplace=True)
data = data.loc[~data.index.duplicated(keep='first')]

# Create features and targets
def create_features_and_targets(data, feature_length):
    X = []
    Y = []

    for i in range(len(data) - feature_length):
        X.append(data.iloc[i:i+feature_length, :].values)
        Y.append(data["Close"].values[i+feature_length])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Function to fetch sentiment scores from StockTwits
def get_sentiment_scores(symbol, start_date, end_date):
    messages = stocktwits.get(symbol, start=start_date, end=end_date)
    sentiment_scores = []

    for message in messages:
        if 'body' in message:
            sentiment_scores.append(sia.polarity_scores(message['body'])['compound'])

    return sentiment_scores

def get_message_volume(symbol, start_date, end_date):
    messages = stocktwits.get(symbol, start=start_date, end=end_date)
    return len(messages)

def get_participation_ratio(symbol, start_date, end_date):
    messages = stocktwits.get(symbol, start=start_date, end=end_date)
    users = set()
    for message in messages:
        if 'user' in message:
            users.add(message['user']['id'])

    participation_ratio = len(users) / len(messages) if len(messages) > 0 else 0
    return participation_ratio

def aggregate_sentiment(symbol, start_date, end_date):
    sentiment_scores = get_sentiment_scores(symbol, start_date, end_date)
    message_volume = get_message_volume(symbol, start_date, end_date)
    participation_ratio = get_participation_ratio(symbol, start_date, end_date)

    # Aggregate sentiment scores
    average_sentiment = np.mean(sentiment_scores)

    # Determine recommendation based on thresholds
    if average_sentiment > 0.75 or message_volume > 75 or participation_ratio > 0.75:
        recommendation = 'Our analysis indicates a potential increase of more than 5%, signaling a Buy recommendation.'
    elif average_sentiment < 0.25 or message_volume < 25 or participation_ratio < 0.25:
        recommendation = 'Based on our projections, a decrease is anticipated, indicating a Sell recommendation.'
    else:
        recommendation = 'Considering the current market conditions, we advise maintaining positions with a Hold recommendation.'

    return recommendation

def calculate_metrics(predictions, actual):
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    return mape, rmse, mae, r2

# Evaluate Models and Calculate Metrics
mape_lstm, rmse_lstm, mae_lstm, r2_lstm = calculate_metrics(predictions_lstm, actual)
# Evaluate Models
predictions_lstm_scaled = lstm_model.predict(Xtest_scaled)
predictions_lstm = target_scaler.inverse_transform(predictions_lstm_scaled)
actual = target_scaler.inverse_transform(Ytest_scaled.reshape(-1, 1))


# Calculate Metrics
mape_lstm, rmse_lstm, mae_lstm, r2_lstm = calculate_metrics(predictions_lstm, actual)

return predictions_lstm, predictions_lr, actual, mape_lstm, rmse_lstm, mae_lstm, r2_lstm, mape_lr, rmse_lr, mae_lr, r2_lr

# Function to train and evaluate LSTM and Linear Regression models
def train_and_evaluate_models(Xtrain, Ytrain, Xtest, Ytest):
    # Scale features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    Xtrain_scaled = feature_scaler.fit_transform(Xtrain.reshape(-1, Xtrain.shape[-1])).reshape(Xtrain.shape)
    Xtest_scaled = feature_scaler.transform(Xtest.reshape(-1, Xtest.shape[-1])).reshape(Xtest.shape)
    Ytrain_scaled = target_scaler.fit_transform(Ytrain.reshape(-1, 1)).reshape(-1)
    Ytest_scaled = target_scaler.transform(Ytest.reshape(-1, 1)).reshape(-1)

    # LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.2, input_shape=(32, 2)))
    lstm_model.add(LSTM(256, recurrent_dropout=0.2))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    lstm_model.compile(loss='mse', optimizer=optimizer)

    # Train LSTM Model
    lstm_model.fit(Xtrain_scaled, Ytrain_scaled,
                   epochs=150,
                   batch_size=128,
                   verbose=0,
                   validation_data=(Xtest_scaled, Ytest_scaled))

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(Xtrain.reshape(-1, Xtrain.shape[-1]), Ytrain)

    # Evaluate Models
    predictions_lstm_scaled = lstm_model.predict(Xtest_scaled)
    predictions_lstm = target_scaler.inverse_transform(predictions_lstm_scaled)
    actual = target_scaler.inverse_transform(Ytest_scaled.reshape(-1, 1))

    predictions_lr = lr_model.predict(Xtest.reshape(-1, Xtest.shape[-1]))

    return predictions_lstm, predictions_lr, actual

# Split data into training and testing sets
feature_length = 32
test_length = 100
X, Y = create_features_and_targets(data, feature_length)
Xtrain, Xtest, Ytrain, Ytest = X[:-test_length], X[-test_length:], Y[:-test_length], Y[-test_length:]

# Train and evaluate LSTM and Linear Regression models
predictions_lstm, predictions_lr, actual = train_and_evaluate_models(Xtrain, Ytrain, Xtest, Ytest)

# Fetch sentiment scores for the same time period as the stock price data
start_date_sentiment = data.index[-test_length:].strftime("%Y-%m-%d")[0]
end_date_sentiment = data.index[-1].strftime("%Y-%m-%d")
sentiment_scores = get_sentiment_scores("^GSPC", start_date_sentiment, end_date_sentiment)

# ========================================== Plotting predicted data ======================================

pred_dict = {"Date": [], "Prediction": []}
for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
        pred_df = pd.DataFrame(pred_dict)
        pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
        pred_fig.update_xaxes(rangeslider_visible=True)
        pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
        plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

        # ========================================== Display Ticker Info ==========================================

        ticker = pd.read_csv('app/Data/Tickers.csv')
        to_search = ticker_value
        ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                        'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
        for i in range(0,ticker.shape[0]):
            if ticker.Symbol[i] == to_search:
                Symbol = ticker.Symbol[i]
                Name = ticker.Name[i]
                Last_Sale = ticker.Last_Sale[i]
                Net_Change = ticker.Net_Change[i]
                Percent_Change = ticker.Percent_Change[i]
                Market_Cap = ticker.Market_Cap[i]
                Country = ticker.Country[i]
                IPO_Year = ticker.IPO_Year[i]
                Volume = ticker.Volume[i]
                Sector = ticker.Sector[i]
                Industry = ticker.Industry[i]
                break

    # ========================================== Page Render section ==========================================
    

        return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                'confidence' : confidence,
                                                'forecast': forecast,
                                                'sentiment': sentiment,  # Include sentiment in context
                                                'action': action,  # Include action in context
                                                'ticker_value':ticker_value,
                                                'number_of_days':number_of_days,
                                                'plot_div_pred':plot_div_pred,
                                                'Symbol':Symbol,
                                                'Name':Name,
                                                'Last_Sale':Last_Sale,
                                                'Net_Change':Net_Change,
                                                'Percent_Change':Percent_Change,
                                                'Market_Cap':Market_Cap,
                                                'Country':Country,
                                                'IPO_Year':IPO_Year,
                                                'Volume':Volume,
                                                'Sector':Sector,
                                                'Industry':Industry,
                                                'mape_lstm': mape_lstm,
                                                'rmse_lstm': rmse_lstm,
                                                'mae_lstm': mae_lstm,
                                                'r2_lstm': r2_lstm,
                                                })

