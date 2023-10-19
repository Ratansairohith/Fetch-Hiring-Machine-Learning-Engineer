import xgboost as xgb
import gradio as gr
import pickle
import pandas as pd

# Fucntion to create lag features

def create_lag_features(df, lags = [1,2,3]):
    df = df.copy()
    for lag in lags:
        df['Receipt_Count_lag' + str(lag)] = df['Receipt_Count'].shift(lag)
    return df

# utils function to extarct date time features
def get_time_feature(df, col, keep=True):
    df_copy = df.copy()
    prefix = col + "_"
    df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    df_copy[prefix + 'weekofyear'] = df_copy[col].dt.isocalendar().week.astype(int)
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek.astype(int)
    df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 4
    df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
    df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
    df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
    if keep: return df_copy
    else: return df_copy.drop([col], axis=1)


def predict_for_year(df, model, n_lags=3):
    # If no pre-trained model is supplied, initialize and train one
    if model is None:
        model = xgb.XGBRegressor(n_estimators=1000, objective='reg:squarederror')
        X_train = df.drop(['Receipt_Count'], axis=1)
        y_train = df['Receipt_Count']
        model.fit(X_train.values, y_train.values)
    
    # Preparing dataset for 2022
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions_2022 = pd.DataFrame(date_rng, columns=['Date'])
    predictions_2022 = get_time_feature(predictions_2022, "Date")
    predictions_2022 = predictions_2022[['Date', 'Date_is_wknd', 'Date_quarter', 'Date_is_month_start', 'Date_is_month_end']]
    
    for ind,day in enumerate(range(len(predictions_2022))):
        for i in range(1, n_lags+1):
            if day-i < 0:
                predictions_2022.loc[day, f'Receipt_Count_lag{i}'] = df.iloc[-(i-day)]['Receipt_Count']
            else:
                predictions_2022.loc[day, f'Receipt_Count_lag{i}'] = predictions_2022.loc[day-i, 'Receipt_Count']
        #print(predictions_2022.head())
        if ind == 0:
          X_test = predictions_2022.iloc[day].drop('Date').values.reshape(1, -1)
        else:
          X_test = predictions_2022.iloc[day].drop(['Date','Receipt_Count']).values.reshape(1, -1)
        #print(X_test.shape)
        pred = model.predict(X_test)[0]
        predictions_2022.loc[day, 'Receipt_Count'] = pred

    return predictions_2022

# Load the trained model

# Define a prediction function to be used with Gradio
def predict_for_month(month_num, show_yearly_trend=False):
    month_num = int(month_num)
    with open("xgb_model.pkl", "rb") as model_file:
      loaded_model = pickle.load(model_file)
    df = pd.read_csv('data_daily.csv')
    df.rename(columns={'# Date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if month_num < 1 or month_num > 12:
        return "Please input a valid month number (1-12)"

    # Predict receipts for the entire year
    predictions_2022 = predict_for_year(df,loaded_model)
    
    # Filter predictions for the selected month only
    start_date = f"2022-{month_num}-01"
    if month_num == 12:
        end_date = "2022-12-31"
    else:
        end_date = f"2022-{month_num+1}-01"
    monthly_predictions = predictions_2022.loc[(predictions_2022['Date'] >= start_date) & (predictions_2022['Date'] < end_date)]
    
    # Sum the predictions to get total receipt count for the month
    total_receipts = monthly_predictions['Receipt_Count'].sum()

    if show_yearly_trend:
        import plotly.express as px
        import io
        from PIL import Image

        fig = px.line(predictions_2022, x='Date', y='Receipt_Count', title='Receipt Predictions for 2022')

        # Convert the plotly figure to an image
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        
        return total_receipts, image   # Return the image along with the total_receipts

    else:
        return total_receipts, None
    
    return total_receipts

# Define Gradio interface
def gr_interface():
    interface = gr.Interface(
        fn=predict_for_month,
        inputs=["number", gr.components.Checkbox(label="Show Yearly Trend")],  # Add a checkbox for the user to choose if they want to see the yearly trend
        outputs=["number","image"],
        title="Receipt Predictions for 2022",
        description="Predict the total receipt count for a month in 2022. Input a month number (1-12)."
    )
    interface.launch(debug=True)

if  __name__ == "__main__":
    gr_interface()