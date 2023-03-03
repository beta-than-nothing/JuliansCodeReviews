import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats 
from sklearn.linear_model import LinearRegression

### Sinn und Zweck dieses Skripts ist es, auf verschiedene Zeitintervalle Linearregressionen durchzuführen. 
# Ich bennene Sachen gerne in Englisch, ist nur meine Angewohnheit.
# Zunächst lade ich die Daten aus einem CSV in einen Dataframe.

df_raw = pd.read_csv('C:\\"Dein Pfad"\\all_UEN_TJ+Months.csv')

# print(df_raw) - falls du dir den Dataframe des csv anschauen willst

# Dieser Abschnitt ordnet und pivotiert den Dataframe - damit bildet die X-Achse die Monate November bis Oktober ab (Reihenfolge des Tourismusjahr) 
# und die Y-Achse des Dataframe die Jahre - bspw. 2008 = November 2007 bis Oktober 2008, nur zum Verständnis
df_pivoted = df_raw.pivot(index = "Tourismus Jahr", columns = "Monat Name")["Gemeinde Anzahl Übernachtungen"]

df_pivoted_ordered_TJ = df_pivoted.iloc[:, [9, 2, 4, 3, 8, 0, 7, 6, 5, 1, 11, 10]] # TJ steht für Tourismusjahr, falls mal eine andere Anordnung gewünscht ist, behalte ich so die Übersicht

# hier extrahiere ich die arrays für die Tourismusjahre und den zu analysierenden Monat, bspw. Januar
extracted_TJ = df_raw["Tourismus Jahr"].tolist()
extracted_month = df_pivoted_ordered_TJ["Januar"].tolist()

# Da im CSV die Jahre ca. 180 mal vorkommen, entferne ich hier die Duplikate
TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

# Hier gebe ich die Listen in zwei neue Variablen, ist vermutlich redundant.
years = TJ 
nights = extracted_month # die Übernachtungen werden (warum auch immer) als Float geladen

# hier wandle ich die Floats in Ints um.
for i in range(len(nights)):
    nights[i] = int(nights[i])

# Diese Liste bestimmt später die Intervalle in Jahren auf Basis derer die Linearregessionen durchgeführt werden
# bspw. 2008-2019 oder 2010-2019, etc.
intervals = []

# Die letzten 4 Werden in den Listen werden hier entfernt (sind die Corona-Jahre und damit oftmals Extremwerte).
if len(years) > 12:
    for i in range(4):
        years.pop(-1)

# Dieser Loop reduziert den Intervall immer um das "älteste" Jahre, also erst schneidet er 2008 ab, dann 2009, 2010, usw.
counter = len(years)
for i in range(len(years)):
    while counter >= 2: # für eine Linearregression brauchen wir ja mindestens 2 Werte, daher stoppt der Counter bei 2.
        intervals.append(years[0:])
        years.pop(0)
        counter -= 1

# Die alten Listen werden von diesem Loop überschrieben, daher ziehe ich sie hier neu.
TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

years = TJ 

# print("Das sind die Jahresintervalle:") # Ausführen zum kontrollieren, falls es dich interessiert
# print(intervals)

# Selbes Spiel nochmal für die Übernachtungen.
nights_intervals = []

# Corona-Jahre raus
if len(years) > 12:
    for i in range(4):
        nights.pop(-1)

### Diese Funktion wird nur gebraucht, falls der ausgewählte Monat einen "NaN" Wert enthält!
### Das kommt vor, wenn wir z.B. den Wert für Mai 2023 suchen würden, den gibt es aber noch nicht.
### Wird aktuell (Analyse ohne Corona-Jahre) nicht gebraucht, ist nur ein Platzhalter für weitere Analyse mit den Corona-Jahre        
### if np.isnan(extracted_month[-1]): 
###    extracted_month.pop(-1) # hier sollte wohl TJ.pop(-1) inkludiert werden, damit die arrays gleich lang sind!

# Die Zeilen 82-103 sind wie für die Jaher oben.
counter = len(nights)
for i in range(len(nights)):
    while counter >= 2:
        nights_intervals.append(nights[0:])
        nights.pop(0)
        counter -= 1

extracted_month = df_pivoted_ordered_TJ["Januar"].tolist()

TJ = []
for i in range(len(extracted_TJ)):
    if extracted_TJ[i] not in TJ:
        TJ.append(extracted_TJ[i])

years = TJ 
nights = extracted_month

for i in range(len(nights)):
    nights[i] = int(nights[i])

# print("Das sind die Übernachtungswerte:") # Ausführen zum kontrollieren, falls es dich interessiert
# print(nights_intervals)

# Die "intervals" und "night_intervals" werden immer kürzer, vor ursprünglich 12, bis letztendlich auf 2 Werte in einer Sublist

total_results = []
# Hier werden die Endergebnisse der einzelnen Linearregressionen zwischengespeichert als ndarrays

# Dieser Loop erstellt die eigentlichen ndarrays für die Linearregressionen
for i in range(len(intervals)):
    years_slice = intervals[i]         # Auf den ersten Blick ist dieser Code Block etwas lang, aber das muss so sein!
    nights_slice = nights_intervals[i] # Jede Sublist wird erst hier nochmals extrahiert.
    input_data = []        
    for item in range(len(years_slice)):
        input_data.append([years_slice[item], nights_slice[item]]) # Und hier in den Zwischenspeicher "input_data" gegeben. 


    linreg_years = np.array(input_data)[:,0].reshape(-1,1) # dann werden daraus die ndarrays gemacht
    linreg_nights = np.array(input_data)[:,1].reshape(-1,1)
    
    predict_years = [2023]  # das Zielparameter festgelegt (Jahr 2023)
    predict_years = np.array(predict_years).reshape(-1,1)

    regsr = LinearRegression() # und hier aus Basis der aus dem Zwischenspeicher generierten ndarrays die Linearregression ausgeführt
    regsr.fit(linreg_years, linreg_nights)
    predicted_y = regsr.predict(predict_years)
    total_results.append(predicted_y) # abschließend wird für jeden Zwischenspeicher das Ergebnis seiner Linearregression in diese Liste eingefügt.
      
# for i in range(len(total_results)): # hier siehst du dann das Ergebnis der in Summe 12 aufeinanderfolgenden Linearregressionen.
#     print(total_results[i]) # der Datentyp sind ndarrays - ich weiß ich wiederhole mich oft.

# Abschließend wandle ich das Ergebnis nochmal in ein "lesbareres" Format um.
flat_result = [a for sublist in total_results for a in sublist] # Hier gehen die Sublisten (jeweils Länge 1) in eine übergeordnete Liste über (immer noch als ndarray)

result = []
for i in range(len(flat_result)):
    result.append(flat_result[i])

result = [x[0] for x in result] # Hier werden die ndarrays dann als Einzelwerte neu in die Liste "result" final eingefügt, ohne Sublisten, etc.

# Die Floats wandle ich nochmal in Ints um.
for i in range(len(result)):
    result[i] = int(result[i])

# Und schaue mir aus Neugier den Durchschnitt an.
SumOfResult = sum(result)
SumOfResult = int(SumOfResult)
average = SumOfResult / len(result)

# hier as Endergebis:
print() # nur zur besseren Lesbarkeit
print("Durchschnitt aller Regressionen:")
print(int(average))
print()
print("Liste der einzelnen Regressionen:")
print(result)
print()

### Abschlussbemerkung: 
# Diese Methode funktioniert ganz gut für die bisherigen Prognosen die ich gemacht habe.
# Ich nehmen dann noch die Daten inkl. der Corona-Jahre her und wende dann die klassische T-Statistik auf den Mittelwert der beiden Analysen an.
# Die Konfidenzintervalle (95%) die ich abschließende erhalte treffen bisher immer das tatsächliche Ergebnis der Tourismusstatistik.