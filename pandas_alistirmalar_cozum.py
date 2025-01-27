# 1:  Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız

import pandas as pd
import seaborn as sns
from git.compat import win_encode

df = sns.load_dataset('titanic')
df.head()

# Görev 2:  Titanic verisetindeki kadın ve erkek yolcuların sayısını bulunuz
df['sex'].value_counts()

# Görev3:  Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.nunique()
df['sex'].unique() # Bu bir değişken içerisindeki uniqe olan değişkenleri bir array olarak verir.

# Görev4:  pclass değişkeninin unique değerlerinin sayısını bulunuz.
df['pclass'].nunique() # değerlerin sayısı
df['pclass'].unique() # değerlerin kendisi

# Görev5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
df[['pclass','parch']].nunique()

#   Görev6:  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df['embarked'].dtype
df['embarked'] = df['embarked'].astype('category')
df['embarked'].dtype
df.info()

# Görev7:  embarked değeri C olanların tüm bilgelerini gösteriniz
pd.set_option('display.max_columns', None)
df[df['embarked'] == 'C'].head()

# Görev8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df['embarked'] != 'S']

#  Görev9:   Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df['age'] < 30) &(df['sex'] == 'male')]

# Görev10:  Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df['fare'] > 500)  | (df['age'] > 70)]

# Görev 11:  Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()    # df.isnull().values.any() Hiç eksik değer var mı?

# Görev 12:  who değişkenini dataframe’den çıkarınız.
df.drop('who', axis=1)

#  Görev13:  deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

df['deck'].mode()
df.isnull().sum()
df['deck'] = df['deck'].fillna(df['deck'].mode()[0]) # mode() bir pandas series döndürür. bu yüzden serinin [0] elemanını aldık.

#  Görev14:  age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df['age'].median() # seri küçükten büyüğe doğru sıralandığında seriyi ortadan ikiye ayıran değer.
df['age'] = df['age'].fillna(df['age'].median())
df.isnull().sum()

#  Görev15:  survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(['pclass', 'sex']).agg({'survived': ['sum', 'count', 'mean']})

# Görev16:  30 yaşınaltında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
#  setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

df['age_flag'] = df['age'].apply(lambda x : 1 if x < 30 else 0)

# Görev17:  Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız

import pandas as pd
import seaborn as sns
df = sns.load_dataset('tips')
df.head()

# Görev18:  Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby('time')['total_bill'].agg(['min', 'max', 'mean'])

# Görev19:  Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby(['time', 'day'])['total_bill'].agg(['min', 'max', 'mean'])

#  Görev 20:  Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
filtered_data = df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')]
filtered_data.groupby(df['day'],).agg({'total_bill':['min', 'max', 'mean'],
                           'tip': ['min', 'max', 'mean']})
df.groupby(['day', 'time']).size() # buradan bakıldığında çumartesi ve pazar günleri lunch'ta hiç veri yoktur.


#  Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
new_df = df.loc[(df['size'] < 3) & (df['total_bill'] >10)]
new_df[['total_bill', 'tip','size']].mean()

(df.loc[(df['size'] < 3) & (df['total_bill'] >10)]).mean(numeric_only=True)


#  Görev22:  total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df['total_bill_tip_sum'] = df['total_bill'] + df['tip']

#  Görev23:  total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
new_df = df.sort_values('total_bill_tip_sum')[:30]
new_df.shape

























