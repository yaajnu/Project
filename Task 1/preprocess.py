import pandas as pd
import category_encoders as ce
import os
import pickle
def get_data():
    return pd.read_csv(os.path.join(os.getcwd(),'cars_price.csv'))
df_car= get_data()
df_car=df_car.drop('Unnamed: 0',axis=1)
df_dummy=df_car.copy()
cols=['make','model']
co=pd.get_dummies(df_dummy['drive_unit']).columns
df_dummy[co]=pd.get_dummies(df_dummy['drive_unit'])
co_seg=pd.get_dummies(df_dummy['segment']).columns
df_dummy[co_seg]=pd.get_dummies(df_dummy['segment'])
co_cond=pd.get_dummies(df_dummy['condition']).columns
df_dummy[co_cond]=pd.get_dummies(df_dummy['condition'])
co_fuel=pd.get_dummies(df_dummy['fuel_type']).columns
df_dummy[co_fuel]=pd.get_dummies(df_dummy['fuel_type'])
co_trans=pd.get_dummies(df_dummy['transmission']).columns
df_dummy[co_trans]=pd.get_dummies(df_dummy['transmission'])
co_color=pd.get_dummies(df_dummy['color']).columns
df_dummy[co_color]=pd.get_dummies(df_dummy['color'])
df_dummy=df_dummy.drop(['priceUSD','fuel_type','color','condition','drive_unit','segment','transmission'],axis=1)
encoder = ce.BackwardDifferenceEncoder(cols=cols)
_=encoder.fit_transform(df_dummy)
df=df_car.drop(['make','model'],axis=1)
with open('variables', 'wb') as f:
    pickle.dump([co,co_seg,co_color,co_cond,co_fuel,co_trans,encoder], f)
