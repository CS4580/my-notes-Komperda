import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("blood_pressure.csv")

df[['bp_before','bp_after']].describe()


df[['bp_before', 'bp_after']].plot(kind='box')
plt.show()


#checking for normal distribution:
df['bp_difference'] = df['bp_before'] - df['bp_after']

df['bp_difference'].plot(kind='hist', title= 'Blood Pressure Difference Histogram')
plt.show()


result = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(result)
