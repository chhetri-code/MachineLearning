import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    """
    Step1: Reads the cytology data
    Step2: Drops the id and Unnamed: 32 columns as they are not required
    Step3: Encodes diangnosis column values as 1 for Malignant and 0 for Benign
    """
    data = pd.read_csv("/Workspace/Repos/chhetri.code@gmail.com/MachineLearning/breast-cancer-detector/data/data.csv")
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    
    return data

def create_model(data): 
    """
    Step1: Creates the X dataset (with features) and y dataset with classification labels
    Step2: Scales the X dataset using StandardScaler
    Step3: Splits the X and y datasets into training (80%) and test datasets (20%)
    Step4: Trains the Logistic Regression model
    Step5: Tests the model and prints the accuracy and classification report
    """


    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test model
    y_pred = model.predict(X_test)
    print('Model Accuracy: ', accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    return model, scaler

def main():
    """
    Main function to run the notebook
    Step1: Clean the data
    Step2: Get the model and scalar objects
    Step3: Create the model and scalar pickle dumps
    """

    data = get_clean_data()
    
    model, scaler = create_model(data)
    
    with open('/Workspace/Repos/chhetri.code@gmail.com/MachineLearning/breast-cancer-detector/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('/Workspace/Repos/chhetri.code@gmail.com/MachineLearning/breast-cancer-detector/model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
