# Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df = pd.read_csv('titanicdataset/titanic_data.csv')
df_pre = df.copy()
df_pre.head
