
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from keras.models import load_model

def main():
    # load the label encoder used during training
    label_encoder = joblib.load('models/FoamAmV1/label_encoder.joblib')

    # load the saved models from a file
    svm_model = joblib.load('models/FoamAmV1/svm_model.joblib')
    ann_model = load_model('models/FoamAmV1/ann_model.h5')
    rf_model = joblib.load('models/FoamAmV1/rf_model.joblib')
    knn_model = joblib.load('models/FoamAmV1/knn_model.joblib')

    y_test = [
        'Baseline',
        'F',
        'I'
    ]

    print(f"y_test: {y_test}")

    X_test = [
        [1.4870, 0.8010, 2.1068, 1.2142, 0.5912],
        [1.6985, 1.1423, 2.2672, 2.0116, 1.7038],
        [1.7512, 3.8921, 2.7285, 3.9591, 2.2637],
    ]

    # make predictions using the loaded models
    svm_y_pred = svm_model.predict(X_test)
    svm_y_pred = label_encoder.inverse_transform(svm_y_pred)
    print(f"SVM: {svm_y_pred}")

    ann_y_pred = np.round(ann_model.predict(X_test)).astype(int).flatten()
    ann_y_pred = label_encoder.inverse_transform(ann_y_pred)
    print(f"ANN: {ann_y_pred}")
    
    rf_y_pred = rf_model.predict(X_test)
    rf_y_pred = label_encoder.inverse_transform(rf_y_pred)
    print(f"RF: {rf_y_pred}")

    knn_y_pred = knn_model.predict(X_test)
    knn_y_pred = label_encoder.inverse_transform(knn_y_pred)
    print(f"KNN: {knn_y_pred}")


main()