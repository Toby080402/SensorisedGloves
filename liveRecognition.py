import time
from statistics import mean
from statistics import variance

import serial
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from matplotlib import style
import numpy as np
import random
import serial
import pandas as pd
import keyboard

from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from keras.models import load_model


def main():
    print("Setting up")

    # Define arduino port
    arduino_port = "COM6"
    # arduino_port = "/dev/cu.usbmodem11401"

    labels = ["pinky", "ring", "middle", "index", "thumb"]

    # Define baud rate (usually 9600)
    baud = 9600

    # Connecting to serial
    ser = serial.Serial(arduino_port, baud)


    # load the label encoder used during training
    label_encoder = joblib.load('models/FlexiAm/label_encoder.joblib')

    # load the saved models from a file
    rf_model = joblib.load('models/FlexiAm/rf_model.joblib')



    # Continually try getting data
    while True:
        # Get and parse bytes from serial port
        ser.flushInput()
        bytes = ser.readline()

        data = bytes.decode('utf-8')
        # print(data)

        # Clean string
        data_trimmed = data[:-2]
        data_trimmed = data_trimmed.replace(" ", "")

        # Split into seperate values (for each finger)
        values = data_trimmed.split(',')

        # print(values)

        # Sometimes will start reading at wrong place so ensure you have 5 values
        if len(values) != 5:
            continue

        # Checks for each value
        for j in range(0, 5):
            if len(values[j]) != 4:
                continue

            try:
                # Check if value can be converted to a float
                current_value = float(values[j])

                # Check if within reasonable bounds (not spike)
                if current_value > 6 or current_value < 0:
                    continue

            except ValueError:
                continue

        # CHECKS OVER

        X_test = [
            values
        ]


        rf_y_pred = rf_model.predict(X_test)
        rf_y_pred = label_encoder.inverse_transform(rf_y_pred)
        print(f"RF: {rf_y_pred}")


        # Clear port to remove delay issue
        ser.flushInput()
        time.sleep(0.001)

    ser.close()


main()
