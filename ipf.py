import os
import numpy as np

def writeData(X,route):
    f = open(route, "w")
    for row in X:
        for x in row[:-1]:
            f.write(str(x) + ",")
        f.write(str(row[-1]) + "\n")
    f.close()

def readData(route_data, route_labels):
    X = []
    f = open(route_data,"r")
    for line in f:
        row = np.array([])
        for l in line.split(" "):
            row = np.append(row,float(l))
        X.append(row)
    f.close()

    y = []
    f = open(route_labels, "r")
    for line in f:
        y.append(str(line.strip()).replace("\"", ""))
    f.close()
    return np.array(X), np.array(y)

def IPF(X,y):
    Xy = np.concatenate((X, y.reshape(-1,1)), axis=1)
    writeData(Xy,"/tmp/data_for_r.txt")
    os.system("Rscript ipf.R")
    cleanX, cleanY = readData("/tmp/data_for_python.txt", "/tmp/labels_for_python.txt")
    #os.system("rm data_for_r.txt data_for_python.txt labels_for_python.txt")
    return cleanX, cleanY
