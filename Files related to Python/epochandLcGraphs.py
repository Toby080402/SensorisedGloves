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
plt.rcParams['font.family'] = 'Helvetica'

def main():
    # plt.style.use('science')
    # # Set the default font family to DejaVu Sans

    #
    # # Set the default font size to 10
    # plt.rcParams['font.size'] = 10
    #
    # # Set the default line width to 0.5
    # plt.rcParams['lines.linewidth'] = 0.5
    #
    # # Set the default marker size to 4
    # plt.rcParams['lines.markersize'] = 4
    #
    # # Set the default figure size to 8x6 inches
    # plt.rcParams['figure.figsize'] = [8.0, 6.0]
    # load the label encoder used during training

    # load the saved models from a file
   # ann_model = tf.keras.models.load_model('model3/FoamCombo/ann_model.h5')

    # Load the training history
    history = joblib.load('model3/FoamCombo/ann_model_history.joblib')

    plt.plot(history['loss'])
    plt.title('Effect of increasing epoch on model accuracy', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # Limit the x-axis to the range 0 to 1000
    plt.xlim([0, 1000])

    # Limit the y-axis to the range 0 to 1
    #plt.ylim([0, 1])
  #  plt.savefig("model3/FoamCombo/images/epochs.png, dpi=300")
    plt.show()

def learning():
    data = pd.read_csv("data/Normalised/FoamAm.csv")
    print(data)
    X = data.iloc[:, 1:].values
    y = data["symbol"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Load the .joblib model
    model = load('model2/FoamAmV2/rf_model.joblib')
    # Define the cross-validation and scoring metric for learning_curve
    cv = 4
    scoring = 'accuracy'

    # Generate the learning curves
    train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=cv, scoring=scoring,
                                                             train_sizes=np.linspace(0.1, 1.0, 11))
    # Add (0, 0) point at the beginning
    # train_sizes = np.insert(train_sizes, 0, 0)
    # train_scores = np.insert(train_scores, 0, [0], axis=1)
    # valid_scores = np.insert(valid_scores, 0, [0], axis=1)

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
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Learning Curve for Foam AmSL RF(2)')
    plt.xlim([8, 84])  #96, 84, 72 single, 180, 157.5, 135
    plt.ylim([0, 1.2])
    plt.savefig("images/learncurves/RF.png", dpi=300)
    plt.show()

def ann():

    # Load the training history
    history = joblib.load('model2/FlexiAr/ann_model_history.joblib')

    # Plot the training and validation loss
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(history['accuracy'], label='Training accuracy')
    plt.plot(history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Accuracy vs. Epochs')
    plt.title('Learning Curve for FlexiForce ArSL ANN(2)')
    plt.xlim([0, 1000])  # 96, 84, 72 single, 180, 157.5, 135
    plt.ylim([0, 1.2])
    plt.savefig("images/learncurves/flexiAr.png", dpi=300)
    plt.show()

#main()
learning()
#ann()