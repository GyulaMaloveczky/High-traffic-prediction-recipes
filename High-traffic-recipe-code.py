# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np

# Loading dataset from URL
url = "https://s3.amazonaws.com/talent-assets.datacamp.com/recipe_site_traffic_2212.csv"
df = pd.read_csv(url)
print(df.shape)

# Data validation
## Recipe column validation
print(df['recipe'].dtype)
print(df['recipe'].duplicated().sum())
print(df['recipe'].isna().sum())

## Calories column validation
print(df['calories'].dtype)
print(df['calories'].isna().sum()/len(df)*100)
df = df.dropna(subset = ['calories'])

## Carbohydrate column validation
print(df['carbohydrate'].dtype)
print(df['carbohydrate'].isna().sum()/len(df)*100)

## Sugar column validation
print(df['sugar'].dtype)
print(df['sugar'].isna().sum()/len(df)*100)

## Protein column validation
print(df['protein'].dtype)
print(df['protein'].isna().sum()/len(df)*100)

## Category column validation and transformation
print(df['category'].dtype)
df.loc[df['category'] == 'Chicken Breast', 'category'] = 'Chicken'
print(set(df['category']))
df['category'] = df['category'].astype('category')
print(df['category'].isna().sum()/len(df)*100)

## Servings column validation and transformation
print(set(df['servings']))
print((len(df[df['servings'] == '4 as a snack']) + len(df[df['servings'] == '6 as a snack']))/len(df)*100)
df = df.drop(df[(df['servings'] == '4 as a snack') | (df['servings'] == '6 as a snack')].index)
df['servings'] = df['servings'].astype('int')
print(df['servings'].dtype)
print(df['servings'].isna().sum()/len(df)*100)

## High_traffic column validation
df['high_traffic'].fillna('No', inplace=True)
print(df['high_traffic'].dtype)

# Data visualization
sns.countplot(data=df, x='high_traffic')
plt.show()

# Histograms for numerical columns
sns.histplot(data=df, x='calories')
plt.show()
sns.histplot(data=df, x='carbohydrate')
plt.show()
sns.histplot(data=df, x='sugar')
plt.show()
sns.histplot(data=df, x='protein')
plt.show()

# Countplots for categorical columns
sns.countplot(data=df, x='category')
plt.xticks(rotation=45)
plt.show()
sns.countplot(data=df, x='servings')
plt.show()

# Heatmap for category vs high_traffic
cross_tab = pd.crosstab(df['category'], df['high_traffic'])
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

# Data preprocessing
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
scaler = MinMaxScaler()
df['calories'] = scaler.fit_transform(df[['calories']])
df['carbohydrate'] = scaler.fit_transform(df[['carbohydrate']])
df['sugar'] = scaler.fit_transform(df[['sugar']])
df['protein'] = scaler.fit_transform(df[['protein']])

# Setting up features and target variable
X = df[['calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings']]
y = df['high_traffic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)

# Decision Tree Model
tree = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}
grid_search = GridSearchCV(tree, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

# Using best parameters to train and test the model
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=3, min_samples_split=10)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Decision Tree model metrics
accuracy_tree = accuracy_score(y_test, y_pred)
print("Accuracy tree:", accuracy_tree)
precision_tree = precision_score(y_test, y_pred, pos_label="High")
print("Precision Score Tree:", precision_tree)
f1_tree = f1_score(y_test, y_pred, pos_label="High")
print("F1 Score tree:", f1_tree)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict
