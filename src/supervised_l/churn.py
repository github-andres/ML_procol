import numpy as np
import matplotlib.pyplot as plt

def load_dataset(link):
    import pandas as pd
    dataset = pd.read_csv(link, index_col=0)
    return dataset

def knn_model(n_neighbors, np_array):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier     
    
    X = np_array[:, :-2]
    y = np_array[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, stratify=y)
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    
    y_prediction = knn.predict(X_test)
    manual_accuracy = (y_prediction == y_test).sum()/len(y_test)
    
    accuracy_test = knn.score(X_test, y_test)
    accuracy_train  = knn.score(X_train, y_train)
    return np.array([[accuracy_test, accuracy_train]])
    
    
if __name__ == "__main__":
    link = "/home/linuxand/jupyter_notebooks/numpy_c/datasets/supervised_l/telecom_churn_clean.csv"
    
    churn_df = load_dataset(link)
    churn_np = churn_df.values
    
    
    for _ in range(100):
        plt.ylim((0.8, 1))
        storage_array_test = np.empty((0, 3))
        
        
        for num_of_neigbours in (range(1, 26)):
            temporary_array = knn_model(num_of_neigbours, churn_np)
            storage_array_test = np.concatenate([storage_array_test, np.concatenate([np.array([[num_of_neigbours]]), temporary_array], axis=1)])
            
        plt.plot(storage_array_test[:, 0], storage_array_test[:, 1], "b-", alpha=0.2)
        plt.plot(storage_array_test[:, 0], storage_array_test[:, 2], "g-", alpha=0.2)
    
    
    plt.legend()
    plt.show()
