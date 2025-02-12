import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

diabetes_df = pd.read_csv("/home/linuxand/jupyter_notebooks/numpy_c/datasets/supervised_l/diabetes_clean.csv")

def regression_model_2d(dataframe, dependent_v, target):
    
    X = dataframe[dependent_v].values.reshape((-1,1))
    y = dataframe[target].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    linear_pred = reg.predict(np.arange(start=diabetes_df["glucose"].values.min(),stop=diabetes_df["glucose"].values.max(), step=.5).reshape(-1,1)) 
    
    return linear_pred, X, y

reg_line, X, y = regression_model_2d(diabetes_df, ["bmi"], ['glucose'])

plt.title("BMI as Glucose explanatory variable")
plt.plot(np.arange(start=X.min(), stop=X.max(), step=0.5), reg_line, "r-", label="Linear Regression")
plt.scatter(X, y, alpha=0.1)
plt.legend()
plt.show()

    





if __name__ == "__main__":
    pass