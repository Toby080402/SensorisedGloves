import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from joblib import load

from keras.models import load_model

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import scienceplots

from sklearn.model_selection import learning_curve


def main():
    plt.style.use('science')
    # Set the default font family to DejaVu Sans
    plt.rcParams['font.family'] = 'Helvetica'

    # Set the default font size to 10
    plt.rcParams['font.size'] = 10

    # Set the default line width to 0.5
    plt.rcParams['lines.linewidth'] = 0.5

    # Set the default marker size to 4
    plt.rcParams['lines.markersize'] = 4

    # Set the default figure size to 8x6 inches
    plt.rcParams['figure.figsize'] = [8.0, 6.0]
    # load the label encoder used during training

    # load the saved models from a file
   # ann_model = tf.keras.models.load_model('model3/FoamCombo/ann_model.h5')

    # Load the training history
    history = joblib.load('model3/FoamCombo/ann_model_history.joblib')

    plt.plot(history['accuracy'])
    plt.title('Effect of increasing epoch on model accuracy', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # Limit the x-axis to the range 0 to 1000
    plt.xlim([0, 1000])

    # Limit the y-axis to the range 0 to 1
    plt.ylim([0, 1])
    plt.savefig("model3/FoamCombo/images/epochs.png", dpi=300)
    plt.show()

def learning():
    data = pd.read_csv("data/Normalised/FoamAm.csv")
    print(data)
    X = data.iloc[:, 1:].values
    y = data["symbol"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Load the .joblib model
    model = load('model3/FoamAmV2/knn_model.joblib')

    # Define the cross-validation and scoring metric for learning_curve
    cv = 4
    scoring = 'accuracy'

    # Generate the learning curves
    train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=cv, scoring=scoring,
                                                             train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate mean and standard deviation for the scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # Plot the learning curves
    plt.plot(train_sizes, train_mean, label='Training ' + scoring)
    plt.plot(train_sizes, valid_mean, label='Validation ' + scoring)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='b')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color='r')

    plt.xlabel('Number of training examples')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.title('Learning Curves')
    plt.show()
    #plt.savefig("images/learncurves/1.png", dpi=300)

#main()
learning()