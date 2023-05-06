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


def main():
    def save_date(iteration):
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
            return iteration

        # Checks for each value
        for j in range(0, 5):
            if len(values[j]) != 4:
                return iteration

            try:
                # Check if value can be converted to a float
                current_value = float(values[j])

                # Check if within reasonable bounds (not spike)
                if current_value > 6 or current_value < 0:
                    return iteration

            except ValueError:
                return iteration

        # Add x and y to lists
        xs.append(iteration)
        for j in range(0, 5):
            ys[j].append(float(values[j]))

        # Clear port to remove delay issue
        ser.flushInput()
        iteration += 1
        time.sleep(0.001)

        return iteration

    print("Setting up")

    # Define arduino port
    arduino_port = "COM6"
    # arduino_port = "/dev/cu.usbmodem11401"

    labels = ["pinky", "ring", "middle", "index", "thumb"]

    # Define baud rate (usually 9600)
    baud = 9600

    # Define file name to save to
    file_title = "PinkyTest"
    data_file_name = f"{file_title}.csv"
    # image_file_name = f"{file_title}.png"


    symbol = 'b'


    unicode = hex(ord(symbol))

    print(f"Testing {symbol} which has a hex unicode: {unicode}")

    print(f"Almost ready to save to:{data_file_name}")

    number_of_iterations = 1
    FRAMES_NUM = 300

    average_ys = [[], [], [], [], []]
    var_ys = [[], [], [], [], []]

    # Run saving data for each iteration
    for k in range(1, number_of_iterations + 1):
        # Waits for user to press enter
        input(f"Press enter to start recording {k} of {number_of_iterations}")

        print(f"Getting record number {k} of {number_of_iterations}")

        # Connecting to serial
        ser = serial.Serial(arduino_port, baud)
        ser.flushInput()

        # Configure plots
        xs = []  # store trials here (n)
        ys = [[], [], [], [], []]

        i = 0

        pbar = tqdm(total=FRAMES_NUM)

        while i < FRAMES_NUM:
            try:
                newi = save_date(i)

                if newi != i:
                    pbar.update(1)
                    i = newi

            except Exception as e:
                print("If you see this, something went wrong")
                print(e)

        pbar.close()

        print(f"Calculating average values from the {len(ys[0])} frames of data")
        if len(ys[0]) <= 0:
            print(f"Messed up on {k} of {number_of_iterations}")
            return

        newRow = [xs, ys[0], ys[1], ys[2], ys[3], ys[4]]

        for j in range(0, 5):
            average_ys[j].append(mean(ys[j]))
            var_ys[j].append(variance(ys[j]))

        print(f"Saving average to file:{data_file_name}")
        # Open our existing CSV file in append mode
        # Create a file object for this file
        with open(data_file_name, 'a') as f_object:

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = csv.writer(f_object)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(xs)
            writer_object.writerow(ys[0])
            writer_object.writerow(ys[1])
            writer_object.writerow(ys[2])
            writer_object.writerow(ys[3])
            writer_object.writerow(ys[4])

            # Close the file object
            f_object.close()

        # Set up new plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Draw x and y lists
        ax.clear()
        for j in range(0, 5):
            ax.plot(xs, ys[j], label=labels[j])

        print("Here is the plot, close the plot window to continue...")
        plt.show()

        # Close serial port and fig
        plt.close()
        ser.close()

        print(f"Finished iteration {k}")


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Draw x and y lists
    ax.clear()
    #for j in range(0, 5):
        #ax.plot(range(1, number_of_iterations+1), average_ys[j], label=labels[j])

    plt.plot(xs, ys[0], label=labels[0])
    plt.ylim(0, 5)
    plt.xlabel("Measurement Number, Time * 8 (s * 8)")
    plt.ylabel("Voltage (V)")
    plt.title("Signal to noise ratio test (Pinky)")
    plt.savefig("pinkyrepro.jpg")
    plt.figure()
    plt.plot(xs, ys[1], label=labels[1])
    plt.ylim(0, 5)
    plt.xlabel("Measurement Number, Time * 8 (s * 8)")
    plt.ylabel("Voltage (V)")
    plt.title("Signal to noise ratio test (Ring)")
    #plt.savefig("1.jpg")
    plt.figure()
    plt.plot(xs, ys[2], label=labels[2])
    plt.ylim(0, 5)
    plt.xlabel("Measurement Number, Time * 8 (s * 8)")
    plt.ylabel("Voltage (V)")
    plt.title("Signal to noise ratio test (Middle)")
    #plt.savefig("2.jpg")
    plt.figure()
    plt.plot(xs, ys[3], label=labels[3])
    plt.ylim(0, 5)
    plt.xlabel("Measurement Number, Time * 8 (s * 8)")
    plt.ylabel("Voltage (V)")
    plt.title("Signal to noise ratio test (Index)")
    #plt.savefig("FoamTest.jpg")
    plt.figure()
    plt.plot(xs, ys[4], label=labels[4])
    plt.ylim(0, 5)
    plt.xlabel("Measurement Number, Time * 8 (s * 8)")
    plt.ylabel("Voltage (V)")
    plt.title("Signal to noise ratio test (Thumb)")
    #plt.savefig("4.jpg")



    print(f"Here is the plot of all {number_of_iterations} averages, close the plot window to continue...")

    plt.show()


    plt.close()

    print(f"Finished all {number_of_iterations} iterations")


main()
