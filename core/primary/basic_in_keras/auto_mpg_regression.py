import tensorflow as tf
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = tf.keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(file_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv(file_path, names=column_names,
                 na_values="?", comment='\t',
                 sep=" ", skipinitialspace=True)

print(df.head())
print(df.isna().sum())

sb.pairplot(df, vars=column_names)
plt.show()

sb.distplot(df["MPG"])
plt.show()

sb.boxplot(y="MPG", x="Cylinders", data=df)
plt.show()
# mpg = df["MPG"]
# cylinders = df["Cylinders"]
#
# print(cylinders)
# sb.distplot(cylinders, kde=False)
# plt.show()
