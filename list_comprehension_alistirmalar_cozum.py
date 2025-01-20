import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.info()

# GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.

df.columns = ["NUM_"+col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]



#GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.

#df.columns = [col if "no" in col.lower() else col+"_FLAG" for col in df.columns]

df.columns = [col.upper()+"_FLAG" if "no" not in col else col.upper() for col in df.columns]


# Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

og_list = ['abbrev', 'no_previous']

new_cols = [col for col in df.columns if col not in og_list]
df[new_cols]


#### Ekstra Alıştırmalar ####

#1. Bir listede 1'den 10'a kadar olan sayıların karesini al
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [num**2 for num in numbers]

#2. Bir stringdeki harfleri büyük harfe çevir
text = "Bir stringdeki harfleri büyük harfe çevir"
upper_list = [char.upper() for char in text]

#3. İç içe listelerdeki tüm öğeleri düz bir listeye çevir
nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
new_list = [item for sublist in nested_list for item in sublist ]

#4. Bir listedeki sayıları 3'e veya 5'e bölünebiliyorsa karesini al. Değilse listeye alma.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_list2 = [item**2 for item in numbers if item %3 ==0 or item %5 ==0]