import os
import time
import random


if __name__ == '__main__':

    Y = []
    X = []
    with open("D:\\Github\\PreprocessedData\\data\\040034411201.txt") as f:
        line = f.readline()
        print(len(line.split(",")[-1].split(" ")))
        for key, value in enumerate(line.split(",")[-1].split(" ")):
            # print(value)
            X.append(key)
            Y.append(float(value.rstrip("\n")))
    print(X, "\n", Y)
