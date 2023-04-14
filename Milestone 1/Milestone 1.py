from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import datetime as dt
import pandas as pd

# apply ordinal Encoding
def ord_Encoding(df, col_name, values_list):
    return df.replace(col_name,values_list)


# get days from date
def date_to_numeric(df, colmns):
    for col in colmns:
        current_date = dt.date.today()
        date = df[col].replace('/', '-', regex=True)
        date_list = (current_date - ((pd.to_datetime(date, format='%d-%m-%Y')).dt.date))
        df[col] = date_list.dt.days
    return df

# read data as csv file
data = pd.read_csv("games-regression-dataset.csv")

## check number of unique vqlues in each columns
# print(len(df[""].unique().tolist()))

# Remove Row that has no value in target column
data = data.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)

Y = data["Average User Rating"]

# drop columns
cols = ["URL", "ID", "Name", "Description", "Subtitle", "Icon URL", "Average User Rating", "Primary Genre"]
X = data.drop(columns=cols)

# apply ordinal encoding
values_list = {"4+": 4, "9+": 3, "12+": 2, "17+": 1}
X = ord_Encoding(X, "Age Rating", values_list)

# convert date columns to numerical data
date_cols =['Original Release Date', 'Current Version Release Date']
X = date_to_numeric(X, date_cols)



print(X)
print(X.isna().sum())