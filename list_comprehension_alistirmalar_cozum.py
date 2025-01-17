import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
# GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.

df.columns = [col.upper() for col in df.columns]


#GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.


#df.columns = [col+"_FLAG" if "no" not in col else col for col in df.columns] BURAYI SOR !!!!!!!!!

df.columns = [col if "no" in col.lower() else col+"_FLAG" for col in df.columns]


# Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

og_list = ['abbrev', 'no_previous']

new_cols = [col for col in df.columns if col not in og_list]
df[new_cols]

