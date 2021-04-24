import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    df = load_data()
    df1 = pre_pro_data(df)
    df2 = df.copy()
    df2 = df2.drop(columns = ['stn', 'wban', 'temp_count', 'dewp_count', 'slp_count', 'stp_count', 'visib_count', 'wdsp_count'])
    
    page = st.sidebar.selectbox("Choose a page", ["Raw Data", "Pre-processed Data", "Visualization", "Correlation Matrix"
                                                 , "Random Forest Regression"])

    if page == "Raw Data":
        st.header("Prediction Analysis for Weather Data")
        st.title("Raw Dataset")
        st.write("Please select a page on the left.")
        st.write(df)
        
    elif page == "Pre-processed Data":
        st.title("Pre-processed Data")
        st.write(df1)
    
    elif page == "Visualization":
        st.title("Data Exploration")
        x_axis = st.selectbox("Choose a variable for the x-axis", df1.columns, index=0)
        y_axis = st.selectbox("Choose a variable for the y-axis", df1.columns, index=1)
        visualize_data(df1, x_axis, y_axis)
        
    elif page == "Correlation Matrix":
        st.title("Correlation Matrix")
        corr_matrix(df2)
        
    elif page == "Random Forest Regression":
        st.title("Random Forest Regression")
        rfr_model(df1, df2)

@st.cache

def load_data():
    df = pd.read_csv('weather_data.csv')
    return df

def pre_pro_data(df):
    dataset = df.copy()
    dataset = dataset.drop(columns = ['stn', 'wban', 'temp_count', 'dewp_count', 'slp_count', 'stp_count', 'visib_count', 'wdsp_count'])
    dataset['timestamp'] = dataset[dataset.columns[0:3]].apply(lambda x: '-'.join(x.dropna().astype(str)),axis=1)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset = dataset.drop(columns = ['year', 'month', 'day'])
    dataset = dataset[['timestamp', 'dewp', 'slp', 'stp', 'visib', 'wdsp',
                       'maxtemp', 'mintemp', 'prcp', 'sndp', 'frshtt']]
    return dataset

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_line().encode(
        x=x_axis,
        y=y_axis
    ).interactive()

    st.write(graph)
    
def corr_matrix(df):
    plt.figure(figsize=(16, 6))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    st.pyplot()
    
def rfr_model(df1, df2):
    #df1 for timestamp column
    #df2 for model
    df2 = df2.drop(columns = ['temp', 'mxspd', 'GUST'])
    train = df2[365:]
    test = df2[:365]
    X_train, y_train = train.drop(columns = ['maxtemp', 'mintemp']), train[['maxtemp', 'mintemp']]
    X_test, y_test = test.drop(columns = ['maxtemp', 'mintemp']), test[['maxtemp', 'mintemp']]
    sc = MinMaxScaler(feature_range=(0,1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    test_set_r2 = r2_score(y_test, pred)
    st.write("RMSE Score: ", test_set_rmse, " R-Squared Value: ",test_set_r2)
    
    predDF = pd.DataFrame(pred, columns=['maxtemp_pred', 'mintemp_pred'])
    output = df1[:365]
    output = output.drop(columns = ['dewp', 'slp', 'stp', 'visib', 'wdsp', 'prcp', 'sndp', 'frshtt'])
    output['maxtemp_pred'] = predDF['maxtemp_pred']
    output['mintemp_pred'] = predDF['mintemp_pred']
    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize = (16,10))
    fig.suptitle('Real vs Predicted Temperature')
    fig.text(0.5, 0.04, 'Timestamp', ha='center')
    fig.text(0.04, 0.5, 'Temperature', va='center', rotation='vertical')
    axs[0].plot(output['timestamp'], output['mintemp'], label = "Real Minimum Temperature")
    axs[0].plot(output['timestamp'], output['mintemp_pred'], label = "Predcited Minimum Temperature")
    axs[0].legend()
    axs[1].plot(output['timestamp'], output['maxtemp'], label = "Real Maximum Temperature")
    axs[1].plot(output['timestamp'], output['maxtemp_pred'], label = "Predcited Maximum Temperature")
    axs[1].legend()
    st.pyplot()

if __name__ == "__main__":
    main()