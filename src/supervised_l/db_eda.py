import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load dataset
def pull_dataset(link):
    diabetes_df = pd.read_csv(link)
    return diabetes_df

# Setting up the traning and testing datasets


def dataset_split(percentage = int, pd_dataset = pd.DataFrame):
    dataset_values = pd_dataset.values
    np.random.shuffle(dataset_values)
    
    limits = int(pd_dataset.shape[0] * percentage)

    training_X_df = dataset_values[:limits, :-2]
    training_Y_df = dataset_values[:limits, -1].flatten().astype(np.bool_)
    testing_X_df = dataset_values[limits:, :-2] 
    testing_Y_df = dataset_values[limits:, -1].flatten()

    return training_X_df, training_Y_df, testing_X_df, testing_Y_df

# Model definition
def model_executing(num_neighbors = 10, training_X_df = np.ndarray, training_Y_df = np.ndarray, testing_X_df = np.ndarray, testing_Y_df = np.ndarray):

    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(training_X_df, training_Y_df)

    y_predicted = knn.predict(testing_X_df)
    accuracy_table = np.concatenate([testing_Y_df.reshape((testing_Y_df.shape[0],1)), y_predicted.reshape((y_predicted.shape[0],1))], axis=1)
    accuracy = (np.where(y_predicted == testing_Y_df, 1, 0).sum()) / accuracy_table.shape[0]
    
    return accuracy

if __name__ == "__main__":    
    link = "/home/linuxand/jupyter_notebooks/numpy_c/datasets/supervised_l/diabetes_clean.csv"
    diabetes_df = pull_dataset(link=link)
    
    diabetes_df["diabetes"] = diabetes_df["diabetes"].astype("boolean")
    diabetes_df.columns = ['pregnancies', 'glucose', 'diastolic', 'triglyceride', 'insulin', 'bmi', 'dpf', 'age', 'diabetes']

    perc_list = [0.7, 0.75, 0.8, 0.85]
    number_of_neighbors = [5, 10, 15, 20, 25, 30]
    storage_3d_array = np.empty((24,3,0))
   
    for _ in range(10):
        storage_array = np.empty((0,3))
        for perc in perc_list:          
            for neig_num in number_of_neighbors:
                training_X_df, training_Y_df, testing_X_df, testing_Y_df = dataset_split(perc, diabetes_df)
                accuracy_ind = model_executing(num_neighbors=neig_num, training_X_df=training_X_df, training_Y_df=training_Y_df, testing_X_df=testing_X_df, testing_Y_df=testing_Y_df)
                array_to_be_concatenated = np.array([[perc, neig_num, accuracy_ind]])
                storage_array = np.concatenate([storage_array, array_to_be_concatenated], axis=0)
        
        storage_array = np.reshape(storage_array, shape=(-1,3,1))
        
        storage_3d_array = np.concatenate([storage_3d_array, storage_array], axis=2)
            
    
    means = storage_3d_array.mean(axis=2)
    np.savetxt("/home/linuxand/jupyter_notebooks/numpy_c/preprocessed/supervised_l/accuracy_table.csv", means, fmt="%f", delimiter=",")
    
        
    

