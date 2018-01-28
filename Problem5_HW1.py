import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("PatientData.csv", header = None)#, header = None)
print(df)

for column in df:
    print(df[column])
    array_of_questions = []
    mean_col = 0
    i = 0
    # try:
    #     mean_col = df[column].mean()
    # except Exception, e:

    for elem in df[column]:
        try:
            elem = float(elem)
        except:
            try:
                elem = int(elem)
            except:
                pass
        if isinstance(elem, basestring):
            # Array of wuaestion is a list of indices
            array_of_questions.append(i)
        else:
            mean_col = mean_col + elem
        i = i +1
    mean_col = mean_col/(i - len(array_of_questions))
    for values in array_of_questions:
        df[column][values] = mean_col
    # i = 0
    # for elem in column:
    #     if elem == "?":
    #         column[i] = mean_col
    #     i = i + 1
#df.shape()
#df.head()

print("GOOD")

