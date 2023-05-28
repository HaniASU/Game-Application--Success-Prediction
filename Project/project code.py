from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.linear_model import RidgeCV,LinearRegression
import numpy as np
import datetime as dt
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time
import warnings

warnings.filterwarnings("ignore")



paths = ["preprocessing/", "models/"]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        os.replace(path, path)


# Convert date to numerical data and handle missing values
def date_to_numeric(df, columns):
    for col in columns:
        current_date = dt.date.today()
        date = df[col].replace('/', '-', regex=True)
        date_list = (current_date - (pd.to_datetime(date, format='%d-%m-%Y')).dt.date)
        df[col] = date_list.dt.days
    return df


# Visualize data in specific column
def visualize_data(y, x, columnx, columny):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(x[columnx], y)
    ax.set_xlabel(columnx)
    ax.set_ylabel(columny)
    plt.show()


# draw distribution of column based on another column
def distribution(X, Y, column_x, column_y):
    df = pd.DataFrame(X[column_x])
    df[column_y] = Y
    df.groupby(column_x)[column_y].plot(kind='kde')
    plt.legend(pd.unique(X[column_x]), title=column_x)
    plt.xlabel(column_y)
    plt.ylabel(column_x)


# Plot the regression line
def plot(y_actual, y_predict, model):
    fig, ax = plt.subplots()
    ax.scatter(y_actual, y_predict)
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color="red", lw=4)
    ax.set_title(model)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.show()


# extract outliers then replace outliers value with next or previous value
def fill_outliers(X, column, threshold):
    outliers = np.abs(zscore(X[column])) > threshold
    value = X[column].shift(1).fillna(X[column].shift(-1))
    X.loc[outliers, column] = value
    return X


Null_values_columns = {}


# Apply preprocessing on data
def replace_null_values(X, type='train'):
    date_cols = ['Original Release Date', 'Current Version Release Date']  # Convert date cols into numerical data
    X = date_to_numeric(X, date_cols)

    if (type == 'train'):
        X = fill_outliers(X, 'User Rating Count', 3)  # detect outliers in dat with Z score 5
        X = fill_outliers(X, 'Price', 3)

        languages = X['Languages'].str.split(', ').explode()
        frequent_languages = languages.value_counts(normalize=True, dropna=True)

        Genres = X['Genres'].str.split(', ').explode()
        frequent_Genres = Genres.value_counts(normalize=True, dropna=True)

        Null_values_columns['Languages'] = ', '.join(frequent_languages.index[:1].tolist())
        Null_values_columns['Genres'] = ', '.join(frequent_Genres.index[:1].tolist())
        Null_values_columns['Age Rating'] = X['Age Rating'].value_counts().idxmax()
        Null_values_columns['User Rating Count'] = int(X["User Rating Count"].mode())
        Null_values_columns['Price'] = int(X["Price"].mode())
        Null_values_columns['Original Release Date'] = int(X['Original Release Date'].mean())
        Null_values_columns['Current Version Release Date'] = int(X['Current Version Release Date'].mean())
        Null_values_columns['Size'] = int(X['Size'].mean())
        preprocessing_dict['Null values columns'] = Null_values_columns

    X['Languages'] = X['Languages'].fillna(Null_values_columns['Languages'])
    X['Size'] = X['Size'].fillna(X['Size'].mean())
    X['Genres'] = X['Genres'].fillna(Null_values_columns['Genres'])
    X['Age Rating'] = X['Age Rating'].fillna(Null_values_columns['Age Rating'])
    X['User Rating Count'] = X['User Rating Count'].fillna(Null_values_columns['User Rating Count'])
    X['Price'] = X['Price'].fillna(Null_values_columns['Price'])
    X['Original Release Date'] = X['Original Release Date'].fillna(Null_values_columns['Original Release Date'])
    X['Current Version Release Date'] = X['Current Version Release Date'].fillna(
        Null_values_columns['Current Version Release Date'])

    return X


# Save as pickle file
def save_parameter(name, data):
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(data, file)


# Load pickle file
def load_parameter(path):
    with open(f'{path}.pkl', 'rb') as file:
        return pickle.load(file)


# Apply one hot encoding in data
def one_hot_encoding(X, type='train'):
    for column in ('Languages', 'Genres'):
        encoded_cols = X[column].str.get_dummies(', ')
        X = pd.concat([X, encoded_cols], axis=1)
        X = X.drop(column, axis=1)

    X = pd.get_dummies(X, columns=['Age Rating'])
    if type == 'train':
        save_parameter(f'{paths[0]}encoded_columns', X.columns.tolist())
    else:
        column_names = load_parameter(f'{paths[0]}encoded_columns')
        X = X.reindex(columns=column_names, fill_value=0)

    return X

# Apply Feature Scaling
def features_scaling(X, cols, type='train'):
    if type == 'train':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_model = scaler.fit(X[cols])
        scaled_data = scaler_model.transform(X[cols])
        save_parameter (f'{paths[0]}cols_scalr', scaler_model)
        save_parameter(f'{paths[0]}scaler_model', scaler_model)
    else:
        scaler_model = load_parameter(f'{paths[0]}scaler_model')
        scaled_data = scaler_model.transform(X[cols])
    df = pd.DataFrame(scaled_data, columns=cols)

    for i in df:
        X.loc[:, i] = df[i].tolist()

    return X

def anova(X_features, Y_train=None, type=f_regression, K=None):
    # Anova
    selected_cols = np.std(X_features, axis=0) != 0  # get columns that have zero standard deviation
    X_features = X_features[
        selected_cols.index[selected_cols]]  # remove zero standard deviation columns from train

    selector = SelectKBest(type, k=K)  # apply Anova feature selection with target k =34
    selector.fit(X_features, Y_train)  # fit X and Y on anova selector

    scores = selector.scores_
    pvalues = selector.pvalues_

    selected_cols = selector.get_support(indices=True)  # get indices of selection columns

    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(len(scores)), -np.log10(pvalues))
    plt.xticks(np.arange(len(scores)), np.arange(1, len(scores) + 1))
    plt.xlabel('Features')
    plt.ylabel('Average User Rating')
    plt.show()

    return (X_features.columns[selected_cols]).tolist()


# Apply Regression features selection using Anova and Correlation on data
def features_selection_regression(X, y_train=None, type='train', numerical_features=None, categorical_features=None):
    if type == 'train':

        # Correlation
        numerical_features.insert(loc=len(numerical_features.columns), column='Average User Rating', value=y_train.tolist())
        corr = numerical_features.corr()                                                                                     # Get the correlation between the features
        correlation_cols = corr.index[abs(corr['Average User Rating']) > 0.05]                                                    # Top 50% Correlation training features with the Value

        plt.subplots(figsize=(12, 8))                                                                                        # Correlation plot
        top_corr = numerical_features[correlation_cols].corr()
        sns.heatmap(top_corr, annot=True)
        plt.show()
        correlation_cols = (correlation_cols.delete(-1)).tolist()

        # Anova
        anova_cols = anova(categorical_features, Y_train=y_train, type=f_regression, K=34)

        cols_selection = anova_cols + correlation_cols
        save_parameter(f'{paths[0]}selected_regression_columns', cols_selection)

    else:
        cols_selection = load_parameter(f'{paths[0]}selected_regression_columns')

    X = X[cols_selection]                            # Choose Features columns selection from data

    return X

# Apply Classification features selection using Anova and Mutual information on data
def features_selection_classification(X, y_train=None, type='train', numerical_features=None, categorical_features=None):
    if type == 'train':
        # Anova
        anova_cols = anova(numerical_features, Y_train=y_train,type=f_classif, K=4)

        # Mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=34)
        selector.fit_transform(categorical_features, y_train)
        selected_cols = selector.get_support(indices=True)                               # get indices of selection columns
        mutual_cols = (categorical_features.columns[selected_cols]).tolist()
        cols_selection = anova_cols + mutual_cols
        save_parameter(f'{paths[0]}selected_classification_columns', cols_selection)

    else:
        cols_selection = load_parameter(f'{paths[0]}selected_classification_columns')

    X = X[cols_selection]                            # Choose Features columns selection from data

    return X

# Apply Regression or Classification
def task_type(X, Y_train=None, type_technique='regression', type_data='train', dict=None, Numerical_cols=None):

    if type_data == 'train':

        X = replace_null_values(X, type=type_data)
        save_parameter(f'{paths[0]}preprocessing_dict', dict)
        X = one_hot_encoding(X, type=type_data)
        Categorical_cols = [x for x in X.columns.tolist() if x not in Numerical_cols]
        X = features_scaling(X, Numerical_cols, type=type_data)

        if type_technique == 'regression':
            X = features_selection_regression(X, Y_train, type=type_data,
                                              numerical_features=X[Numerical_cols],
                                              categorical_features=X[Categorical_cols])

        elif type_technique == 'classification':
            X = features_selection_classification(X, Y_train, type=type_data,
                                                  numerical_features=X[Numerical_cols],
                                                  categorical_features=X[Categorical_cols])

    else:

        X = replace_null_values(X, type=type_data)
        X = one_hot_encoding(X, type=type_data)
        X = features_scaling(X, Numerical_cols, type=type_data)

        if type_technique == 'regression':
            X = features_selection_regression(X, type=type_data)

        elif type_technique == 'classification':
            X = features_selection_classification(X, type=type_data)

    return X

def get_labels(data):
    Labels = {'Low': 1, 'Intermediate': 2, 'High': 3}
    result = []
    for i in data:
        result.append(Labels[i])
    return result


# Preprocessing Dictionary
preprocessing_dict = {}

# Read data as csv file
regression_data = pd.read_csv("datasets/games-regression-dataset.csv")

# Clean Row that has no value in target column
regression_data = regression_data.dropna(axis=0, how="any", subset="Average User Rating", inplace=False)

Y_reg = regression_data["Average User Rating"]


# Get columns that percentage of unique values in each column greater than 55%
info_unique_cols = regression_data.nunique().multiply(other=100 / regression_data.shape[0])
Unique_cols = info_unique_cols.loc[info_unique_cols > 55].index.tolist()
Unique_cols.remove("Size")

# Get columns that percentage of missing values in each column greater than 35%
info_missing_cols = (regression_data.isna().sum() * 100) / len(regression_data)
missing_value_cols = info_missing_cols.loc[info_missing_cols > 35].index.tolist()

# Get columns that they will be dropped
drop_cols = Unique_cols + missing_value_cols            # drop columns that have more unique and missing values
drop_cols.append("Primary Genre")                       # drop (Primary Genre) column because its value is in (Genre) column
drop_cols.append("Average User Rating")
preprocessing_dict['drop columns'] = drop_cols          # Add drop columns to preprocessing dictionary

# Drop some columns from data
X_reg = regression_data.drop(columns=drop_cols)
Numerical_cols = X_reg.select_dtypes(exclude=['object']).columns.tolist() +['Original Release Date','Current Version Release Date']
save_parameter(f'{paths[0]}Numerical_cols', Numerical_cols)

# visualization of some columns based on target column
visualize_data(Y_reg,X_reg,'User Rating Count','Average User Rating')
visualize_data(Y_reg,X_reg,'Age Rating','Average User Rating')
distribution(X_reg,Y_reg,'Age Rating','Average User Rating')
visualize_data(Y_reg,X_reg ,'Price','Average User Rating')


# Visualizing the relationships between features
sns.pairplot(data = X_reg.join(Y_reg), height = 20)



# Spilt data into train and test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, Y_reg, test_size=0.20, shuffle=True, random_state=12)

task = ['regression', 'classification']

# Apply preprocessing in train and test data
X_train_reg = task_type(X_train_reg, Y_train=y_train_reg, type_technique=task[0], type_data='train',
                        dict=preprocessing_dict, Numerical_cols=Numerical_cols)

X_test_reg = task_type(X_test_reg, type_technique=task[0], type_data='test',
                       dict=preprocessing_dict, Numerical_cols=Numerical_cols)




# RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train_reg, y_train_reg)

# Ridge Regression Model
fold = KFold(n_splits=5)
ridge_model = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=fold)
ridge_model.fit(X_train_reg, y_train_reg)

# LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

models = {"RandomForestRegressor":RF, 'Ridge':ridge_model, "LinearRegression":linear_model}

for name in models:
    model = models[name].fit(X_train_reg, y_train_reg)
    save_parameter(f'{paths[1]}{name}', model)
    train_pred_ = model.predict(X_train_reg)
    test_pred_ = model.predict(X_test_reg)
    plot(y_train_reg, train_pred_, str(name))
    plot(y_test_reg, test_pred_, str(name))
    print(f"----------------------------{name} model----------------------------------\n")
    print(f'Mean Square Error of train {name} Model : ', mean_squared_error(y_train_reg, train_pred_))
    print(f'Mean Square Error of test {name} Model : ', mean_squared_error (y_test_reg, test_pred_))
    print(f'R2 Score train {name} Model : ', r2_score(y_train_reg, train_pred_))
    print(f'R2 Score  test {name} Model : ', r2_score(y_test_reg, test_pred_))
    print("\n")

classification_data = pd.read_csv("datasets/games-classification-dataset.csv")
classification_data = classification_data.dropna(axis=0, how="any", subset="Rate", inplace=False)
Y_class = get_labels(classification_data["Rate"].tolist())
drop_cols.remove('Average User Rating')
X_class = classification_data.drop(columns=drop_cols + ['Rate'])


X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, Y_class, test_size=0.20, shuffle=True, random_state=10)

X_train_class = task_type(X_train_class, Y_train=y_train_class, type_technique=task[1], type_data='train', dict=preprocessing_dict,Numerical_cols=Numerical_cols)
X_test_class = task_type(X_test_class, type_technique=task[1], type_data='test', dict=preprocessing_dict,Numerical_cols=Numerical_cols)


scores_list = []
n_neighbors_list = [2,5,10,20,30,40,50,60,70,80,90,100,120,140,180,200]
for n_neighbors_value in n_neighbors_list:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors_value)
    knn.fit(X_train_class, y_train_class)
    scores_list.append(knn.score(X_test_class, y_test_class))

fig, ax = plt.subplots()
ax = sns.lineplot(x=n_neighbors_list, y=scores_list, color="red")
ax.set(xlabel="number neighbors", ylabel="Score Model")
plt.title("KNN Model Score with neighbors hyperparameter", fontsize=16)
plt.show()
training_time = []
testing_time =[]
accuracy = []

# KNN classifier
param_grid = {
    'n_neighbors': [2,5,10,20,30,40,50,60,70,80,90,100,120,140,180,200],
    'metric': ['euclidean', 'manhattan']
}
kNN = KNeighborsClassifier()

grid_search = GridSearchCV(kNN, param_grid, cv=5)
grid_search.fit(X_train_class, y_train_class)

best_params_knn = grid_search.best_params_
print(best_params_knn)
knn = KNeighborsClassifier(**best_params_knn)
start_time_train = time.time()
knn.fit(X_train_class, y_train_class)
end_time_train = time.time()
training_time.append(end_time_train-start_time_train)

save_parameter(f'{paths[1]}{knn}_model', knn)
start_time_testing = time.time()
y_pred = knn.predict(X_test_class)
end_time_testing = time.time()
testing_time.append(end_time_testing-start_time_testing)
knn_accuracy = accuracy_score(y_test_class, y_pred)
print('KNN Testing Accuracy:', knn_accuracy)
accuracy.append(knn_accuracy)



# SVM classifier
# Hyperparameter Tuning
param_grid = {'C': [0.1, 1,5, 10], 'kernel': ['linear', 'rbf']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_class, y_train_class)

best_params = grid_search.best_params_
print(best_params)
svm = SVC(**best_params)

# Model Training and Evaluation
start_time_train = time.time()
svm.fit(X_train_class, y_train_class)
end_time_train = time.time()
training_time.append(end_time_train-start_time_train)
save_parameter(f'{paths[1]}{svm}_model', svm)

start_time_testing = time.time()
y_pred = svm.predict(X_test_class)
end_time_testing = time.time()
testing_time.append(end_time_testing-start_time_testing)

svm_accuracy = accuracy_score(y_test_class, y_pred)
print('SVM Testing Accuracy:', svm_accuracy)
accuracy.append(svm_accuracy)

# DecisionTree Classifier
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10,12,14,16,18,20 ,None]
}

decision_tree = DecisionTreeClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(decision_tree, param_grid, cv=5)
grid_search.fit(X_train_class, y_train_class)

best_params = grid_search.best_params_
print(best_params)
decision_tree = DecisionTreeClassifier(**best_params)

start_time_train = time.time()
decision_tree.fit(X_train_class, y_train_class)
end_time_train = time.time()
training_time.append(end_time_train-start_time_train)

save_parameter(f'{paths[1]}{decision_tree}_model', svm)

start_time_testing = time.time()
y_pred = decision_tree.predict(X_test_class)
end_time_testing = time.time()
testing_time.append(end_time_testing-start_time_testing)

decision_tree_accuracy = accuracy_score(y_test_class, y_pred)
print('Decision tree Testing Accuracy:', decision_tree_accuracy)
accuracy.append(decision_tree_accuracy)



plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.bar(range(len(accuracy)), accuracy)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy')

plt.subplot(1, 3, 2)
plt.bar(range(len(training_time)), training_time)
plt.xlabel('Models')
plt.ylabel('Time (minutes)')
plt.title('Total Training Time')

plt.subplot(1, 3, 3)
plt.bar(range(len(testing_time)), testing_time)
plt.xlabel('Models')
plt.ylabel('Time (minutes)')
plt.title('Total Test Time')

# Show the bar graphs
plt.tight_layout()
plt.show()