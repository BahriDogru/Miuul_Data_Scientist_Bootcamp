### Adım1:   flo_data_20K.csv verisini okuyunuz.

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

from flo_musteri_segmantasyonu_cozum import today_date

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv('datasets/flo_data_20k.csv')
df = df_.copy()


# Adım2:  Aykırı değerleri baskılamak için gerekli olan outlier_thresholdsve replace_with_thresholdsfonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Adım3:  "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız

list_ =['order_num_total_ever_online',
        'order_num_total_ever_offline',
        'customer_value_total_ever_offline',
        'customer_value_total_ever_online']
df.describe().T
for col in list_:
    replace_with_thresholds(df, col)

# Adım4:  Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#  alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df['total_order_num'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_value'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

# Adım5:  Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
df[[col for col in df.columns if 'date' in col]] = df[[col for col in df.columns if 'date' in col]].apply(pd.to_datetime, errors='coerce')


### Görev 2:  CLTV Veri Yapısının Oluşturulması
# Adım1:  Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız
today_date = dt.datetime(2021, 6, 1)
df['last_order_date'].max()
#  Adım2:  customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i
#  oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df =pd.DataFrame()
cltv_df['customer_id'] = df['master_id']
cltv_df['recency_cltv_weekly'] = ((df['last_order_date'] - df['first_order_date']).dt.days )/ 7
cltv_df['T_weekly'] = ((today_date - df['first_order_date']).dt.days) / 7
cltv_df['frequency'] = df['total_order_num']
cltv_df['monetary_cltv_avg'] = df['total_value'] / df['total_order_num']


# Görev 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması

#  Adım1:  BG/NBD modelini fit ediniz.
#  • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve
#  exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#  • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve
#  exp_sales_6_month olarak cltv dataframe'ine ekleyiniz

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit( cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

cltv_df['exp_sales_3_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                       cltv_df['frequency'],
                                                                                       cltv_df['recency_cltv_weekly'],
                                                                                       cltv_df['T_weekly'])


cltv_df['exp_sales_6_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                                       cltv_df['frequency'],
                                                                                       cltv_df['recency_cltv_weekly'],
                                                                                       cltv_df['T_weekly'])


#  Adım2:  Gamma-Gamma modelinifit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip
#  exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

#  Adım3:  6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# • Cltvdeğeri enyüksek20 kişiyi gözlemleyiniz.

cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6)

ggf.customer_lifetime_value(bgf,
                          cltv_df['frequency'],
                          cltv_df['recency_cltv_weekly'],
                          cltv_df['T_weekly'],
                          cltv_df['monetary_cltv_avg'],
                          time=6).sort_values(ascending=False)[:20]


# Görev 4:  CLTV Değerine Göre Segmentlerin Oluşturulması
# Adım1:  6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df['segment'] = pd.qcut(cltv_df['cltv'], 4,labels=['D','C','B','A'])
cltv_df[['cltv','segment','frequency','monetary_cltv_avg']].groupby(['segment']).agg({'count',  'sum'})












