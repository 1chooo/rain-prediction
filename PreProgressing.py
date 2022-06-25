import os
import glob
import pandas as pd

path_domestic = os.path.abspath(os.getcwd()) + '/data'
root = "./data"
data = glob.glob(os.path.join(root, "*.csv"))


dataFrame = pd.concat((pd.read_csv(file) for file in data))

dataFrame.to_csv("./20years_data.csv", encoding = "utf_8_sig", index = False) 