import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(data):
    data = pd.concat([data,pd.get_dummies(data['type'], prefix='type')],axis=1).drop(['type'],axis=1)
    data.drop(['Date', 'Unnamed: 0', 'region'], axis = 1, inplace = True)

    numer_cols = ['year', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    scaler = StandardScaler()
    vals = scaler.fit_transform(data[numer_cols])
    data[numer_cols] = vals
    
    return data