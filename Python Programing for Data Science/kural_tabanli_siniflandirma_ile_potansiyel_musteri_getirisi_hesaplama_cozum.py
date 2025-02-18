
## GÖREV 1

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import pandas as pd
df = pd.read_csv('datasets/persona.csv')
df.info()
df.describe().T

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df['SOURCE'].unique()
df['SOURCE'].value_counts()

# Soru 3:Kaç unique PRICE vardır?
df['PRICE'].nunique()

# Soru 4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df['PRICE'].value_counts()

# Soru 5:Hangi ülkeden kaçar tane satış olmuş?
df['COUNTRY'].value_counts()

# Soru 6:Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby('COUNTRY')['PRICE'].sum()


# Soru 7:SOURCE türlerine göre satış sayıları nedir?
df.groupby('SOURCE').value_counts()


#  Soru 8:Ülkelere göre PRICE ortalamaları nedir?

df.groupby('COUNTRY')['PRICE'].mean()
df.pivot_table('PRICE','COUNTRY')


# Soru 9:SOURCE'laragöre PRICE ortalamaları nedir?

df.groupby('SOURCE')['PRICE'].mean()
df.pivot_table('PRICE','SOURCE')

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.pivot_table('PRICE', ['COUNTRY','SOURCE'])


## GÖREV 2

# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.pivot_table('PRICE',['COUNTRY', 'SOURCE', 'SEX', 'AGE'])
df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE'])['PRICE'].mean()


## GÖREV 3:

# Çıktıyı PRICE’a göre sıralayınız
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = (df.pivot_table('PRICE',['COUNTRY', 'SOURCE', 'SEX', 'AGE'])).sort_values(by='PRICE', ascending=False)


## GÖREV 4:

# Indekste yer alan isimleri değişken ismine çeviriniz.
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.

agg_df.reset_index(inplace=True) # index'teki değeri değişken yapmak

## GÖREV 5:

# Age değişkenini kategorik değişkene çeviriniz ve agg_df’eekleyiniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Bunun için en çok kullanılan iki fonksiyon pd.cut() ve pd.qcut()

agg_df['AGE_CAT'] = pd.cut(agg_df['AGE'], bins=[0,25,35,45,55], labels = ["0_25", "26_35", "36_45", "46_55"])
agg_df.info()


## GÖREV 6:

# Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni eklenecek değişkenin adı: customers_level_based
# Birden fazla oluşan customers_level_based değişkenlerini groupby ile price ortalamalarını al

agg_df['customers_level_based']  = [col[0].upper()+'_'+col[1].upper()+'_'+col[2].upper()+'_'+str(col[5]) for col in agg_df.values]
drop_list = ['COUNTRY', 'SOURCE', 'SEX', 'AGE', 'AGE_CAT']
agg_df.drop(drop_list, axis=1, inplace=True)
# agg_df['customers_level_based'].value_counts()
agg_df = agg_df.groupby('customers_level_based').agg({'PRICE':'mean'})



# Görev 7:

# Yeni müşterileri (personaları) segmentlere ayırınız
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz
# Segmentlere göre group by yapıp price mean, max, sum’larını alınız

# price_mean = pd.DataFrame(agg_df.groupby('customers_level_based')['PRICE'].mean())
# price_mean['SEGMENT'] = pd.qcut(price_mean['PRICE'],4, labels = ['D', 'C', 'B','A'])
# agg_df = pd.merge(agg_df, price_mean[['SEGMENT']],on="customers_level_based")

agg_df['SEGMENT'] = pd.qcut(agg_df['PRICE'],4, labels = ['D', 'C', 'B','A'])
agg_df.groupby('SEGMENT').agg({'PRICE':['mean','max','sum']})

# Görev 8:
# Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_26_35"
agg_df[agg_df['customers_level_based'] == new_user]

new_user = "FRA_IOS_FEMALE_26_35"
agg_df[agg_df['customers_level_based'] == new_user]


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

notlar = [68, 74, 82, 90, 78, 85, 92, 88, 76, 61, 79, 73, 89, 81, 72, 95, 70, 83, 77, 75]

plt.hist(notlar, bins=10, edgecolor='r', alpha=0.7)
plt.xlabel('Notlar')
plt.ylabel('Frekans')
plt.title('Sınav Notları Dağılımı')
plt.show()



###############################################
# EKSTRA ÇALIŞMA
###############################################

# Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


# Step 2-3. Import the dataset and Assign it to a variable called chipo.
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')

# Step 4. See the first 10 entries
chipo.head(10)


# Step 5. What is the number of observations in the dataset?
chipo.shape[0]
chipo.info()

# Step 6. What is the number of columns in the dataset?
chipo.shape[1]

# Step 7. Print the name of all the columns.
[col for col in chipo.columns]

# Step 8. How is the dataset indexed?
chipo.index

# Step 9. Which was the most-ordered item?
x = chipo.groupby('item_name')['quantity'].sum().sort_values(ascending=False)
x.head(1)

# Step 10. For the most-ordered item, how many items were ordered?
x = chipo.groupby('item_name')['quantity'].sum().sort_values(ascending=False)
x.head(1)


# Step 11. What was the most ordered item in the choice_description column?
y = chipo.groupby('choice_description')['quantity'].sum().sort_values(ascending=False)
y.head(1)

# Step 12. How many items were orderd in total?
chipo['quantity'].sum()

# Step 13. Turn the item price into a float
chipo['item_price'] =[col[1:] for col in chipo['item_price']]
chipo['item_price'] = chipo['item_price'].astype(float)

# Step 13.a. Check the item price type
chipo['item_price'].dtype

























