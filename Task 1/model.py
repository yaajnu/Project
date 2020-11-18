import pandas as pd
import numpy as np
import seaborn as sns
import category_encoders as ce
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import streamlit as st
import matplotlib
import os
from sklearn.metrics import mean_squared_error
import pickle
#Test Data uploaded must be of the same format of the original data file(cars_price.csv)
st.title('Task 1')
print(os.path.join(os.getcwd(),'cars_price.csv'))
@st.cache
def get_data():
    return pd.read_csv(os.path.join(os.getcwd(),'cars_price.csv'))
df_car= get_data()
st.write('''
A small preview of the input data we're going to be working with
''')
st.write(df_car.head())
cols=['make','model']
df_car=df_car.drop('Unnamed: 0',axis=1)
#We can drop Unnamed 0 as it is just a copy of the index values 
df=df_car.drop(['make','model'],axis=1)
df_dummy=pd.DataFrame()
df_car=pd.DataFrame()
del df_dummy
del df_car
#Dropping make and model as it makes for a very congested plot 
fig=plt.figure(figsize=(30,30))
cols=3
plt.subplots_adjust(wspace=0.4,hspace=0.4)
rows=int(np.ceil((df.shape[1])/cols))
for i,column in enumerate(df.columns):
  ax=fig.add_subplot(rows,cols,i+1)
  ax.set_title(column)
  if df[column].dtype==np.object:
    sns.countplot(y=column,data=df)
    plt.xticks(rotation=45)
  else:
    sns.distplot(df[column])
    plt.xticks(rotation=45)
df=[]
del df
with open('variables', 'rb') as f:
    co,co_seg,co_color,co_cond,co_fuel,co_trans,encoder=pickle.load(f)
#plt.show()
st.write('A graphical analysis of the data we have ')
st.pyplot(fig)
option=st.selectbox('Enter input method(if csv it should have same format as input)',('Upload csv file','Enter columnwise'))
if option=='Upload csv file':
    file=st.file_uploader("Upload csv file with data")
    if file is not None:
        df_input=pd.read_csv(file)
        df_input['volume(cm3)']=df_input['volume(cm3)'].fillna(0)
        df_input.fillna(method='ffill',inplace=True)
        #converting the input data to one_hot 
        df_input[co]=pd.get_dummies(df_input['drive_unit'].astype(pd.CategoricalDtype(categories=co)))
        df_input[co_seg]=pd.get_dummies(df_input['segment'].astype(pd.CategoricalDtype(categories=co_seg)))
        df_input[co_cond]=pd.get_dummies(df_input['condition'].astype(pd.CategoricalDtype(categories=co_cond)))
        df_input[co_fuel]=pd.get_dummies(df_input['fuel_type'].astype(pd.CategoricalDtype(categories=co_fuel)))
        df_input[co_trans]=pd.get_dummies(df_input['transmission'].astype(pd.CategoricalDtype(categories=co_trans)))
        df_input[co_color]=pd.get_dummies(df_input['color'].astype(pd.CategoricalDtype(categories=co_color)))
        df_labels=df_input['priceUSD']
        df_input.drop(['priceUSD','Unnamed: 0','fuel_type','color','condition','drive_unit','segment','transmission'],axis=1,inplace=True)
        print(df_input)
        df_bd=encoder.transform(df_input)
        df_bd=df_bd.drop(['intercept'],axis=1)
        #Preview of the encoded data 
        st.write(df_bd.head())
        st.write('Shape of the transformed data is',df_bd.shape)
        filename='saved.sav'
        loaded_model=pickle.load(open(os.path.join(os.getcwd(),filename),'rb'))
        y_pred=loaded_model.predict(df_bd)
        st.write("The predictions are as follows {}".format(y_pred))
        st.write("Root mean square Error of prediction comes out to be {}".format(mean_squared_error(y_pred,df_labels,squared=False)))
