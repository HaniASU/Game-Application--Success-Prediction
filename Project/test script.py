import datetime as dt
import pandas as pd
import pickle

paths = ["preprocessing/", "models/"]

def load_parameter(path):
    with open(f'{path}.pkl', 'rb') as file:
        return pickle.load(file)


def date_to_numeric(df, columns):
    for col in columns:
        current_date = dt.date.today()
        date = df[col].replace('/', '-', regex=True)
        date_list = (current_date - (pd.to_datetime(date, format='%d-%m-%Y')).dt.date)
        df[col] = date_list.dt.days
    return df

def one_hot_encoding(X):
    for column in ('Languages', 'Genres'):
        encoded_cols = X[column].str.get_dummies(', ')
        X = pd.concat([X, encoded_cols], axis=1)
        X = X.drop(column, axis=1)

    X = pd.get_dummies(X, columns=['Age Rating'])

    column_names = load_parameter(f'{paths[0]}encoded_columns')
    X = X.reindex(columns=column_names, fill_value=0)
    return X

def features_scaling(X):
    scaler_model = load_parameter(f'{paths[0]}scaler_model')
    scaled_data = scaler_model.transform(X)
    X = pd.DataFrame(scaled_data, columns=X.columns)
    return X

def feature_selection_regression(X):
    final_selected_cols = load_parameter(f'{paths[0]}selected_regression_columns')
    return X[final_selected_cols]
def feature_selection_classification(X):
    final_selected_cols = load_parameter(f'{paths[0]}selected_classification_columns')
    return X[final_selected_cols]


preprocessing_cols = load_parameter(f'{paths[0]}preprocessing_regression')


# Regression task

# Read data as csv file
data_regression = pd.read_csv("games-regression-dataset.csv")

# Clean Row that has no value in target column
data_regression = data_regression.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)

# Get X ad Y for regression
Y_reg = data_regression["Average User Rating"]
X_reg = data_regression.drop(columns=preprocessing_cols['drop columns'])

# Apply preprocessing in X data
for column, value in preprocessing_cols['Null values columns'].items():
    if column in X_reg:
        X_reg[column] = X_reg[column].fillna(value)

date_cols = ['Original Release Date', 'Current Version Release Date']
X_reg = date_to_numeric(X_reg, date_cols)
X_reg = one_hot_encoding(X_reg)
X_reg = features_scaling(X_reg)
X_reg = feature_selection_regression(X_reg)


# Classification task

# Read data as csv file
data_classification = pd.read_csv("datasets/games-classification-dataset.csv")

# Clean Row that has no value in target column
data_classification = data_classification.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)

# Get X ad Y for classification
Y_class = data_classification["Average User Rating"]
X_class = data_classification.drop(columns=preprocessing_cols['drop columns'])

# Apply preprocessing in X data
for column, value in preprocessing_cols['Null values columns'].items():
    if column in X_class:
        X_class[column] = X_class[column].fillna(value)

date_cols = ['Original Release Date', 'Current Version Release Date']
X_class = date_to_numeric(X_class, date_cols)
X_class = one_hot_encoding(X_class)
X_class = features_scaling(X_class)
X_class = feature_selection_classification(X_class)

print(X_class)

