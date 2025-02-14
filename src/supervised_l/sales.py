import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

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

def model_creation(X, y, test_percentage):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
        
    accuracy = reg.score(X_test, y_test)
    X_for_y_predicted = np.arange(X.min(), X.max(), step=100).reshape(-1, 1)
    y_predicted = reg.predict(X_for_y_predicted)
    
    return X_for_y_predicted, y_predicted, accuracy
    
     
if __name__ == "__main__":
    sources_to_be_evaluated = ["Social Media", "Radio"]
    sources = [media_to_np(x) for x in sources_to_be_evaluated]
              
    fig, axs = plt.subplots(1, len(sources), figsize=(8, 4))
    counter = 0
    
    for ad_source in sources:
        X = stack_dependent_variables([ad_source])
        X_for_y_predicted, y_predicted, accuracy  = model_creation(X, y, 0.3)        

        
        axs[counter].scatter(ad_source.flatten(), y, alpha=0.1)
        axs[counter].plot(X_for_y_predicted.flatten(), y_predicted, color="red")
        axs[counter].set_title(f"Influence of {sources_to_be_evaluated[counter]} \n Ads over Sales")
        axs[counter].text(ad_source.max() * .5, 50000, f"R^2= {accuracy:.2f}")
               
             
        counter += 1
    
    plt.tight_layout()
    plt.show()
      


