import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv("ProductPriceIndex.csv")
print(df.head())

print(df.info())
df["date"]=pd.to_datetime(df["date"])

df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["day"]=df["date"].dt.day
print(df.info())


df["averagespread"] = pd.to_numeric(df["averagespread"].replace(r'[%]', '', regex=True), errors='coerce')
df["farmprice"] = pd.to_numeric(df["farmprice"].replace(r'[\$,]', '', regex=True), errors='coerce')
df["atlantaretail"] = pd.to_numeric(df["atlantaretail"].replace(r'[\$,]', '', regex=True), errors='coerce')
df["chicagoretail"] = pd.to_numeric(df["chicagoretail"].replace(r'[\$,]', '', regex=True), errors='coerce')
df["losangelesretail"] = pd.to_numeric(df["losangelesretail"].replace(r'[\$,]', '', regex=True), errors='coerce')
df["newyorkretail"] = pd.to_numeric(df["newyorkretail"].replace(r'[\$,]', '', regex=True), errors='coerce')

le=LabelEncoder()
df["product_en"]=le.fit_transform(df["productname"])
feature_cols = ["farmprice","atlantaretail","chicagoretail",
                "losangelesretail","newyorkretail","product_en"]
df_clean = df[feature_cols + ["averagespread"]].copy()  


df_clean = df_clean.dropna()
x=df_clean[feature_cols]
y=df_clean["averagespread"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knr=DecisionTreeRegressor()
knr.fit(x_train,y_train)

print("Test score",knr.score(x_test,y_test))
print("Train score",knr.score(x_train,y_train))
print("Mean absolute error",mean_absolute_error(y_test,knr.predict(x_test)))
print("Mead square error",mean_squared_error(y_test,knr.predict(x_test)))

param_grid = {
    'max_depth': [None, 5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1],
    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"]
}


rr=RandomizedSearchCV(estimator=DecisionTreeRegressor(),param_distributions=param_grid,n_iter=3)
rr.fit(x_train,y_train)

print(rr.best_params_)
print(rr.best_score_)