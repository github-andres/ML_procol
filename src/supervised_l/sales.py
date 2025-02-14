import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

sys.executable

sales_df = pd.read_csv("/home/linuxand/jupyter_notebooks/ML_procol/datasets/supervised_l/advertising_and_sales_clean.csv")
y = sales_df.iloc[:,-1].values

def media_to_np(media):
    X_tv = sales_df.iloc[:, 0].values.reshape((-1,1))
    X_radio = sales_df.iloc[:, 1].values.reshape((-1,1))
    X_social_media = sales_df.iloc[:, 2].values.reshape((-1,1))
    X_influencer = sales_df.iloc[:, 3].values.reshape((-1,1))

    dict_of_media = {"TV": X_tv,
                    "Radio": X_radio,
                    "Social Media": X_social_media,
                    "Influencers": X_influencer}

    return dict_of_media[media]

def stack_dependent_variables(a_py_list):
    list_to_return = np.concatenate(a_py_list, axis=1)
    return list_to_return

def cross_val_model(X, y, cv_folds=5):
    reg = LinearRegression()
    scores = cross_val_score(reg, X, y, cv=cv_folds)
    return scores

if __name__ == "__main__":
    sources_to_be_evaluated = ["Social Media", "Radio"]
    sources = [media_to_np(x) for x in sources_to_be_evaluated]

    fig, axs = plt.subplots(1, len(sources), figsize=(8, 4))
    counter = 0

    for ad_source in sources:
        X = stack_dependent_variables([ad_source])
        scores = cross_val_model(X, y, cv_folds=5)

        axs[counter].scatter(ad_source.flatten(), y, alpha=0.1)
        axs[counter].set_title(f"Influence of {sources_to_be_evaluated[counter]} \n Ads over Sales")
        axs[counter].text(ad_source.max() * .5, 50000, f"CV Mean R^2= {scores.mean():.2f}")

        counter += 1

    plt.tight_layout()
    plt.show()



