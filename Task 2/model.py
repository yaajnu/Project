import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
#Test Data uploaded must be of the same format of the original data file(propulsion.csv)
st.title('Task 2')
@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv(os.path.join(os.getcwd(),'propulsion.csv'))
df= get_data()
try:
    df.drop('Unnamed: 0',axis=1,inplace=True)
except:
    pass
st.write('''
A small preview of the input data we're going to be working with
''')
st.write(df.head())
scaler=StandardScaler()
_=scaler.fit_transform(df.iloc[:,:-2])
df=[]
#plt.show()
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
st.write('A graphical analysis of the data we have ')
st.image('curves.png',width=1200)
st.write('Enter input data(in csv format)')
file=st.file_uploader("Upload csv file with data")
if file is not None:
    df_input=pd.read_csv(file)
    df_labels=df_input.iloc[:,-2:]
    df_input.drop('Unnamed: 0',inplace=True,axis=1)
    df_input.drop(df_labels.columns,axis=1,inplace=True)
    if df_input.shape[0]==1:
        df_bd=scaler.transform(df_input.values.reshape(1,-1))
        st.write(pd.DataFrame(df_bd,columns=df_input.columns))
    else:
        df_bd=pd.DataFrame(scaler.transform(df_input),columns=df_input.columns)
        st.write(df_bd.head())
    #Preview of the scaled data 
    st.write('Shape of the transformed data is',df_bd.shape)
    #Loading saved model of a random forest regressor saved using pickle
    filename='Randomforest_task2.sav'
    loaded_model=pickle.load(open(os.path.join(os.getcwd(),filename),'rb'))
    y_pred=loaded_model.predict(df_bd)
    st.write("The predictions are as follows")
    st.write(pd.DataFrame(y_pred,columns=df_labels.columns))
    st.write("Root mean square Error of prediction comes out to be {}".format(mean_squared_error(y_pred,df_labels,squared=False)))
