from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LinearRegression,Ridge
import numpy as np
import datetime as dt
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import metrics
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Convert date to numerical data and handle missing values
def date_to_numeric(df, columns):
    for col in columns:
        current_date = dt.date.today()
        date = df[col].replace('/', '-', regex=True)
        date_list = (current_date - ((pd.to_datetime(date, format='%d-%m-%Y')).dt.date))
        df[col] = date_list.dt.days
    return df

# Visualize data in specific column
def visualize_data(y,x,columnx,columny):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(x[columnx],y)
    ax.set_xlabel(columnx)
    ax.set_ylabel(columny)
    plt.show()


# draw distribution of column based on another column
def draw_distribution(X,Y,column_x,column_y):
    df = pd.DataFrame(X[column_x])
    df[column_y] = Y
    df.groupby(column_x)[column_y].plot(kind='kde')
    plt.legend(pd.unique(X[column_x]), title=column_x)
    plt.xlabel(column_y)
    plt.ylabel(column_x)


# extract outliers then replace outliers value with next or previous value
def fill_outliers(X, column, threshold):
    outliers = np.abs(zscore(X[column])) > threshold
    value = X[column].shift(1).fillna(X[column].shift(-1))
    X.loc[outliers, column] = value
    return X


# Apply one hot encoding in data
def one_hot_econcoding(data):
    languages = data['Languages'].str.split(', ').explode()
    frequent_languages = languages.value_counts(normalize=True, dropna=True)

    Genres = data['Genres'].str.split(', ').explode()
    frequent_Genres = Genres.value_counts(normalize=True, dropna=True)

    data['Languages'] = data['Languages'].fillna(','.join(frequent_languages.index[:3].tolist()))
    data['Genres'] = data['Genres'].fillna( ','.join(frequent_Genres.index[:3].tolist()))
    encoded_cols = data['Languages'].str.get_dummies(', ')
    data = pd.concat([data, encoded_cols], axis=1)
    data = data.drop('Languages', axis=1)

    encoded_cols = data['Genres'].str.get_dummies(', ')
    data = pd.concat([data, encoded_cols], axis=1)
    data = data.drop('Genres', axis=1)

    return data


dict_value={}
# Apply preprocessing on data
def preprocessing(data,type = 'test'):

    date_cols = ['Original Release Date', 'Current Version Release Date']                                       # Convert date cols into numerical data
    data = date_to_numeric(data, date_cols)
    if (type == 'train'):

        data = fill_outliers(data, 'User Rating Count', 5)                                                      # detect outliers in dat with Z score 5
        data = fill_outliers(data, 'Price', 5)

        dict_value['User Rating Count'] = data["User Rating Count"].mode()
        dict_value['Price'] = data["Price"].mode()
        dict_value['Original Release Date'] = data['Original Release Date'].mean()
        dict_value['Current Version Release Date'] = data['Current Version Release Date'].mean()
        dict_value['Age Rating'] = data['Age Rating'].value_counts().idxmax()


    data['User Rating Count'] = data['User Rating Count'].fillna(dict_value['User Rating Count'])
    data['Price'] = data['Price'].fillna(dict_value['Price'])
    data['Age Rating'] = data['Age Rating'].fillna(dict_value['Age Rating'])
    data['Original Release Date'] = data['Original Release Date'].fillna(dict_value['Original Release Date'])
    data['Current Version Release Date'] = data['Current Version Release Date'].fillna(dict_value['Current Version Release Date'])

    data = pd.get_dummies(data, columns=['Age Rating'])

    return data

# Apply Features selection and Feature Scaling using Anova on data
def features_selection(data,y_train = None,type ='test'):

    if (type == 'train'):

        dict_value['selected cols'] = np.std(data, axis=0) != 0                                        # get columns that have zero standard deviation
        data = data[dict_value['selected cols'].index[dict_value['selected cols']]]                    # remove zero standard deviation columns from train
        selector = SelectKBest(f_regression, k=34)                                                     # apply Anova feature selection with target k =34
        selector.fit(data, y_train)                                                                    # fit X and Y on anova selector

        scores = selector.scores_
        pvalues = selector.pvalues_

        plt.figure(figsize=(20, 10))
        plt.bar(np.arange(len(scores)), -np.log10(pvalues))
        plt.xticks(np.arange(len(scores)), np.arange(1, len(scores) + 1))
        plt.xlabel('Feature')
        plt.ylabel('Average User Rating')
        plt.show()

        dict_value['cols_selection'] = selector.get_support(indices=True)                              # get indices of selection columns

    if (type == 'test'):
        data = data[dict_value['selected cols'].index[dict_value['selected cols']]]                     # remove zero standard deviation columns from test

    # Choose Features columns selection from data
    data = data.iloc[:, dict_value['cols_selection']]

    # Apply feature scaling in data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    return data


# Read data as csv file
data = pd.read_csv("games-regression-dataset.csv")

# Clean Row that has no value in target column
data = data.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)


Y = data["Average User Rating"]

# Get columns that percentage of unique values in each column greater than 55%
info_unique_cols = data.nunique().multiply(other = 100/data.shape[0])
Unique_cols = (info_unique_cols.loc[info_unique_cols > 55].index).tolist()
Unique_cols.remove("Size")

# Get columns that percentage of missiing values in each column greater than 35%
info_missing_cols = (data.isna().sum() * 100)/ len(data)
missing_value_cols = (info_missing_cols.loc[info_missing_cols > 35].index).tolist()

# Get columns that they will be dropped
drop_cols = Unique_cols + missing_value_cols                                                     # drop columns that have more unique and missing values
drop_cols.append("Primary Genre")                                                                # drop (Primary Genre) column because its value is in (Genre) column
drop_cols.append("Average User Rating")                                                          # drop target column because its X (features)

# Drop some columns from data
X = data.drop(columns=drop_cols)

# Apply one hot encoding in all data
X = one_hot_econcoding(X)

# visualization of some columns based on target column
visualize_data(Y,X,'User Rating Count','Average User Rating')
visualize_data(Y,X,'Age Rating','Average User Rating')
draw_distribution(X,Y,'Age Rating','Average User Rating')
visualize_data(Y,X ,'Price','Average User Rating')


# Visualizing the relationships between features
# sns.pairplot(data = X.join(Y), height = 20)

# Spilt data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

# Apply preprocessing in train and test data
X_train = preprocessing(X_train,'train')
X_test = preprocessing(X_test)
X_train = features_selection(X_train, y_train,'train')
X_test = features_selection(X_test)


# Apply Support Vector Regression Model
svr = SVR(kernel='rbf', epsilon=0.1)
svr.fit(X_train, y_train)
train_pred_svr = svr.predict(X_train)
test_pred_svr = svr.predict(X_test)

# Apply Lasso regression Model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
train_pred_lasso = lasso.predict(X_train)
test_pred_lasso = lasso.predict(X_test)

# Apply Linear Regression Model
linear = LinearRegression()
linear.fit(X_train,y_train)
train_pred_linear = linear.predict(X_train)
test_pred_linear = linear.predict(X_test)

# Apply Ridge Regression Model
ridge = Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
train_pred_ridge = ridge.predict(X_train)
test_pred_ridge= ridge.predict(X_test)

# Plot the regression line
fig, ax = plt.subplots()
ax.scatter(y_train, train_pred_svr)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],color= "red",lw=4)
ax.set_title('Support Vector Regression line train ')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()


print('Mean Square Error of train Ridge Regression Model : ', metrics.mean_squared_error(y_train, train_pred_ridge))
print('Mean Square Error of test Ridge Regression Model : ', metrics.mean_squared_error(y_test, test_pred_ridge))

print('Mean Square Error of train Support Vector Regression  Model : ', metrics.mean_squared_error(y_train, train_pred_svr))
print('Mean Square Error of test Support Vector Regression  Model : ', metrics.mean_squared_error(y_test, test_pred_svr))

print('Mean Square Error of train Lasso Regression  Model : ', metrics.mean_squared_error(y_train, train_pred_lasso))
print('Mean Square Error of test Lasso Regression  Model : ', metrics.mean_squared_error(y_test, test_pred_lasso))

print('Mean Square Error of train Linear Regression Model : ', metrics.mean_squared_error(y_train, train_pred_linear))
print('Mean Square Error of test Linear Regression Model : ', metrics.mean_squared_error(y_test, test_pred_linear))





