import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats 
from sklearn.linear_model import LinearRegression

df_raw = pd.read_csv('E:\\Dokumente\\Programming  and BI Stuff\\Python Shenannigans\\venv\\Scripts\\TW ML Model\\all_UEN_TJ+Months.csv')

# print(df_raw)

########################################## TESTING ##########################################

df_pivoted = df_raw.pivot(index = "Tourismus Jahr", columns = "Monat Name")["Gemeinde Anzahl Übernachtungen"]

df_pivoted_ordered_TJ = df_pivoted.iloc[:, [9, 2, 4, 3, 8, 0, 7, 6, 5, 1, 11, 10]]

extracted_TJ = df_raw["Tourismus Jahr"].tolist()
extracted_month = df_pivoted_ordered_TJ["Februar"].tolist()

TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

if np.isnan(extracted_month[-1]):
    extracted_month.pop(-1)
    TJ.pop(-1)

years = TJ 
nights = extracted_month

intervals = []

counter = len(years)
for i in range(len(years)):
    while counter >= 2:
        intervals.append(years[0:])
        years.pop(0)
        counter -= 1

TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

if np.isnan(extracted_month[-1]):
    extracted_month.pop(-1)
    TJ.pop(-1)

nights_intervals = []

counter = len(nights)
for i in range(len(nights)):
    while counter >= 2:
        nights_intervals.append(nights[0:])
        nights.pop(0)
        counter -= 1

extracted_month = df_pivoted_ordered_TJ["Februar"].tolist()

TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

years = TJ 
nights = extracted_month

if np.isnan(extracted_month[-1]):
    nights.pop(-1)
    nights_intervals.pop(-1)
    intervals.pop(-1)

for i in range(len(nights)):
    nights[i] = int(nights[i])

total_results = []
for i in range(len(intervals)):
    years_slice = intervals[i]         # appending these values is necessary
    nights_slice = nights_intervals[i] # otherwise only the last pair gets stored
    input_data = []        
    for item in range(len(years_slice)):
        input_data.append([years_slice[item], nights_slice[item]])


    linreg_years = np.array(input_data)[:,0].reshape(-1,1) 
    linreg_nights = np.array(input_data)[:,1].reshape(-1,1)
    
    predict_years = [2023]
    predict_years = np.array(predict_years).reshape(-1,1)

    regsr = LinearRegression()
    regsr.fit(linreg_years, linreg_nights)
    predicted_y = regsr.predict(predict_years)
    total_results.append(predicted_y)

flat_result = [a for sublist in total_results for a in sublist]

result = []
for i in range(len(flat_result)):
    result.append(flat_result[i])

result = [x[0] for x in result]

for i in range(len(result)):
    result[i] = int(result[i])

sumOfresult = sum(result)
sumOfresult = int(sumOfresult)
average = sumOfresult / len(result)

print() # nur zur besseren Lesbarkeit
print("Durchschnitt aller Regressionen:")
print(int(average))
print()
print("Liste der einzelnen Regressionen:")
print(result)
print()