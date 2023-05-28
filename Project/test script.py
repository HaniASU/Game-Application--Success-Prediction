import datetime as dt
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

import warnings

warnings.filterwarnings("ignore")
paths = ["preprocessing/", "models/"]

def load_parameter(path):
    with open(f'{path}.pkl', 'rb') as file:
        return pickle.load(file)


def date_to_numeric(df, columns):
    for col in columns:
        current_date = dt.date.today()
        date = df[col].replace('/', '-', regex=True)
        ndate = (pd.to_datetime(date, format='%d-%m-%Y')).dt.date
        date_list = (current_date - ndate)
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


def features_scaling(X, cols):

    scaler_model = load_parameter(f'{paths[0]}scaler_model')
    scaled_data = scaler_model.transform(X[cols])
    df = pd.DataFrame(scaled_data, columns=cols)

    for i in df:
        X.loc[:, i] = df[i].tolist()

    return X

def feature_selection_regression(X):
    final_selected_cols = load_parameter(f'{paths[0]}selected_regression_columns')
    return X[final_selected_cols]

def feature_selection_classification(X):
    final_selected_cols = load_parameter(f'{paths[0]}selected_classification_columns')
    return X[final_selected_cols]

def get_labels(data):
    Labels = {'Low': 1, 'Intermediate': 2, 'High': 3}
    result = []
    for i in data:
        result.append(Labels[i])
    return result


preprocessing_cols = load_parameter(f'{paths[0]}preprocessing_dict')
drop_cols_class = preprocessing_cols['drop columns']

drop_cols = preprocessing_cols['drop columns']
# drop_cols.remove('Rate')
drop_cols_reg = drop_cols + ['Average User Rating']


# Regression task

# Read data as csv file
data_regression = pd.read_csv("datasets/ms1-games-tas-test-v1.csv")

date_cols = ['Original Release Date', 'Current Version Release Date']
X_reg = date_to_numeric(data_regression, date_cols)

# Clean Row that has no value in target column
# data_regression = data_regression.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)

Numerical_cols = load_parameter(f'{paths[0]}Numerical_cols')
# Get X ad Y for regression
Y_reg = data_regression["Average User Rating"]
X_reg = data_regression.drop(columns=drop_cols)

# Apply preprocessing in X data
for column, value in preprocessing_cols['Null values columns'].items():
    if column in X_reg:
        X_reg[column] = X_reg[column].fillna(value)

X_reg = one_hot_encoding(X_reg)
X_reg = features_scaling(X_reg,Numerical_cols)
X_reg = feature_selection_regression(X_reg)

print("-----------------------------Regression-------------------------\n")
RF_model = load_parameter(f"{paths[1]}RandomForestRegressor")


test_pred_ = RF_model.predict(X_reg)
print(f"----------------------------RandomForestRegressor model----------------------------------\n")
print(f'Mean Square Error of test RandomForestRegressor Model : ', mean_squared_error (Y_reg, test_pred_))
print(f'R2 Score  test RandomForestRegressor Model : ', r2_score(Y_reg, test_pred_))
print("\n")


# Classification task

# Read data as csv file
data_classification = pd.read_csv("datasets/ms2-games-tas-test-v1.csv")

# Clean Row that has no value in target column
data_classification = data_classification.dropna(axis=0, how="any", subset="Rate", inplace=False)

# Get X ad Y for classification
Y_class = get_labels(data_classification["Rate"].tolist())
X_class = data_classification.drop(columns=drop_cols_class)

date_cols = ['Original Release Date', 'Current Version Release Date']
X_class = date_to_numeric(X_class, date_cols)
# Apply preprocessing in X data
for column, value in preprocessing_cols['Null values columns'].items():
    if column in X_class:
        X_class[column] = X_class[column].fillna(value)



X_class = one_hot_encoding(X_class)
X_class = features_scaling(X_class,Numerical_cols)
X_class = feature_selection_classification(X_class)


print("-----------------------------Classification-------------------------\n")
svm_model = load_parameter(f"{paths[1]}SVC(C=0.1, kernel='linear')_model")

test_pred_ = svm_model.predict(X_class)
print(f"----------------------------SVC model----------------------------------\n")
print(f'Mean Square Error of test SVC Model : ', accuracy_score (Y_class, test_pred_))
print(len(test_pred_))
print("\n")


