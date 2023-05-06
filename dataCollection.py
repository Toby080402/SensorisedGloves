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
    file_title = "FlexiArSL"
    data_file_name = f"{file_title}.csv"
    # image_file_name = f"{file_title}.png"


    symbol = 'Kaaf'


    #unicode = hex(ord(symbol))
    #unicode='xxx'
    #print(f"Testing {symbol} which has a hex unicode: {unicode}")

    print(f"Almost ready to save to:{data_file_name}")

    number_of_iterations = 10
    FRAMES_NUM = 50

    average_ys = [[], [], [], [], []]
    var_ys = [[], [], [], [], []]

    increment=[]

    # Run saving data for each iteration
    for k in range(1, number_of_iterations + 1):
        # Waits for user to press enter
        input(f"Press enter to start recording {k} of {number_of_iterations}")

        print(f"Getting record number {k} of {number_of_iterations}")

        increment.append(k)

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

        newRow = [symbol, mean(ys[0]), mean(ys[1]), mean(ys[2]), mean(ys[3]), mean(ys[4])]

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
            writer_object.writerow(newRow)

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




    #create x value and bar width
    x = np.arange(number_of_iterations)
    width = 0.35
    #Plot x and y values
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - 2 * width / 5, average_ys[0], width / 5, label=labels[0])
    bar2 = ax.bar(x - width / 5, average_ys[1], width/5, label=labels[1])
    bar3 = ax.bar(x, average_ys[2], width/5, label=labels[2])
    bar4 = ax.bar(x + width / 5, average_ys[3], width/5, label=labels[3])
    bar5 = ax.bar(x + 2 * width / 5, average_ys[4], width / 5, label=labels[4])
    ax.set_ylim(0, 5)
    ax.set_xlabel("Test Number")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("'"+symbol+"' Data Set")
    ax.legend(labels, loc="best")

    #,"Test 3","Test 4","Test 5"
    ###
  #  x = np.arange(5)
  #  width = 0.35
# Plot x and y values
    #fig, ax = plt.subplots()
   # for j in range(0, 4):
      #  bar1 = ax.bar(x - width/2, average_ys[j], width, label=labels[j])
     #   bar2 = ax.bar(x + width/2, average_ys[j], width, label=labels[j])

    #Set x and y labels
    ax.set_xticks(x, increment)
    #plt.savefig(symbol + "s.jpg")
    #ax.set_xticklabels()
    #ax.legend()

    print(f"Here is the plot of all {number_of_iterations} averages, close the plot window to continue...")

    plt.show()


    plt.close()

    print(f"Finished all {number_of_iterations} iterations")


main()
