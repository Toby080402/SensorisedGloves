import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

RANDOM_STATE = 80402


def train_SVM(X, y):
    label_encoder = joblib.load('models/FoamAmV2/label_encoder.joblib')

    # split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    # define the parameter grid to search over
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + [0.1, 1, 10]
    }

    # create an SVM model
    svm_model = SVC()

    # perform grid search to find the best hyperparameters
    print("SVM: Carrying out hyperparameter optimisation")
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # use the best model found by grid search to make predictions on the validation set
    best_model = grid_search.best_estimator_
    y_pred_val = best_model.predict(X_val)

    print(f"y_val: \n{label_encoder.inverse_transform(y_val)}")
    print(f"y_pred_val: \n{label_encoder.inverse_transform(y_pred_val)}")

    # evaluate the best model's accuracy on the validation set
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"SVM: Validation accuracy: {val_accuracy:.4f}")

    # evaluate the best model's accuracy on the test set
    y_pred_test = best_model.predict(X_test)

    print("=========\n")


    print(f"y_test: \n{label_encoder.inverse_transform(y_test)}")
    print(f"y_pred_test: \n{label_encoder.inverse_transform(y_pred_test)}")

    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"SVM: Test accuracy: {test_accuracy:.4f}")

    # save the trained model to a file
    print("SVM: Saving model to file")
    joblib.dump(best_model, 'models/FoamAmV2/svm_model.joblib')

def train_ANN(X, y):
    # split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    # define the ANN model architecture and compile it
    ann_model = Sequential()
    ann_model.add(Dense(64, activation='relu', input_dim=5))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(32, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(16, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(1, activation='sigmoid'))
    ann_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # train the ANN model
    print("ANN: Training Model")
    history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=16, verbose=0)

    # use the trained ANN model to make predictions on the validation set
    y_pred_val = np.round(ann_model.predict(X_val)).astype(int).flatten()


    # evaluate the trained ANN model's accuracy on the validation set
    val_accuracy_ann = accuracy_score(y_val, y_pred_val)
    print(f"ANN: Validation accuracy: {val_accuracy_ann:.4f}")

    # use the trained ANN model to make predictions on the test set
    y_pred_test_ann = ann_model.predict(X_test)
    test_accuracy_ann = accuracy_score(y_test, y_pred_test_ann)
    print(f"ANN: Test accuracy: {test_accuracy_ann:.4f}")

    # save the trained model to a file
    print("ANN: Saving model to file")
    ann_model.save("models/FoamAmV2/ann_model.h5")

def train_RF(X, y):
    # split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    # create a Random Forest classifier and train it on the training data
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)

    # use the trained Random Forest classifier to make predictions on the validation set
    y_pred_val = rf_model.predict(X_val)

    # evaluate the trained Random Forest classifier's accuracy on the validation set
    val_accuracy_rf = accuracy_score(y_val, y_pred_val)
    print(f"RF: Validation accuracy: {val_accuracy_rf:.4f}")

    # use the trained Random Forest classifier to make predictions on the test set
    y_pred_test_rf = rf_model.predict(X_test)
    test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
    print(f"RF: Test accuracy: {test_accuracy_rf:.4f}")

    # save the trained model to a file
    print("RF: Saving model to file")
    joblib.dump(rf_model, 'models/FoamAmV2/rf_model.joblib')

def train_KNN(X, y):
    # split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    # create a KNN classifier and train it on the training data
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # use the trained KNN classifier to make predictions on the validation set
    y_pred_val = knn_model.predict(X_val)

    # evaluate the trained KNN classifier's accuracy on the validation set
    val_accuracy_knn = accuracy_score(y_val, y_pred_val)
    print(f"KNN: Validation accuracy: {val_accuracy_knn:.4f}")

    # use the trained KNN classifier to make predictions on the test set
    y_pred_test_knn = knn_model.predict(X_test)
    test_accuracy_knn = accuracy_score(y_test, y_pred_test_knn)
    print(f"KNN: Test accuracy: {test_accuracy_knn:.4f}")

    # save the trained model to a file
    print("KNN: Saving model to file")
    joblib.dump(knn_model, 'models/FoamAmV2/knn_model.joblib')

def main():
    # read in the data
    data = pd.read_csv("data/FoamAmSL.csv")

    # drop any rows with missing data
    data.dropna(inplace=True)

    # extract the features (Voltages) and target (Symbol)
    X = data.iloc[:, 1:].values
    y = data["symbol"].values

    # one hot encode the symbol labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # save the label encoder to a file for later use
    joblib.dump(label_encoder, 'models/FoamAmV2/label_encoder.joblib')

    train_SVM(X, y)
    #train_ANN(X, y)
    #train_RF(X, y)
    #train_KNN(X, y)


main()