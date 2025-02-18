########### Adım1 #########
# flo_data_20K.csv verisini okuyunuz.Dataframe’inkopyasını oluşturunuz.
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df_ = pd.read_csv('datasets/flo_data_20k.csv')
df = df_.copy()

######## Adım2 #######
#  Veri setinde
#  a. İlk 10 gözlem,
#  b. Değişken isimleri,
#  c. Betimsel istatistik,
#  d. Boş değer,
#  e. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()

######## Adım3 #######
# Omnichannel müşterilerin hem online'dan hemdeoffline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#  alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df['total_number_of_purchases'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_price'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']


######## Adım4 #######
# Değişkentiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz
df.info()
#df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df[[col for col in df.columns if 'date' in col]] = df[[col for col in df.columns if 'date' in col]].apply(pd.to_datetime, errors='coerce')

###### Adım 5 #######
# Adım5:  Alışveriş kanallarındaki müşteri sayısının, toplam
# alınan ürün sayısının ve toplam harcamaların dağılımına bakınız

df.groupby('order_channel').agg({'master_id': 'count',
                                 'total_number_of_purchases': 'sum',
                                 'total_price': 'mean'})


############ Adım 6 #################
# En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby('master_id').agg({'total_price': 'sum',}).sort_values('total_price', ascending=False).head(10)


########## Adım 7 ##################
# En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby('master_id').agg({'total_number_of_purchases': 'sum',}).sort_values('total_number_of_purchases', ascending=False).head(10)


###########  Adım 8 ##########
#  Veri ön hazırlık sürecini fonksiyonlaştırınız.

df.isnull().sum()
df.describe().T
df['interested_in_categories_12'].unique()
df['interested_in_categories_12'].value_counts()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, 'total_number_of_purchases')
replace_with_thresholds(df, 'total_price')

def check_data(dataframe):
    dataframe['total_number_of_purchases'] = dataframe['order_num_total_ever_online'] + dataframe['order_num_total_ever_offline']
    dataframe['total_price'] = dataframe['customer_value_total_ever_offline'] + dataframe['customer_value_total_ever_online']

    dataframe[[col for col in dataframe.columns if 'date' in col]] = (dataframe[[col for col in dataframe.columns if 'date' in col]].
                                                                      apply(pd.to_datetime, errors='coerce'))

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    replace_with_thresholds(dataframe, 'total_number_of_purchases')
    replace_with_thresholds(dataframe, 'total_price')

    return dataframe

df = check_data(df)
df.describe().T
df.info()
### GÖREV 2  RFM Metriklerinin Hesaplanması

# Recency (analizin yapıldığı tarih - müşterinin son satın alma tarihi),
# Frequency (Müşterinin yaptığı toplam satın alma),
# Monetary (Müşterinin bıraktığı toplam para)

df['last_order_date'].max()
# recency i hesaplamak için bir tarih belirlememiz gerekiyor
today_date = dt.datetime(2021,6,1)

rfm = df.groupby('master_id').agg({'last_order_date': lambda lod: (today_date - lod.max()).days,
                                   'total_number_of_purchases': lambda x : x.sum(),
                                   'total_price': lambda x : x.sum()})

rfm.columns = ['recency','frequency','monetary']


### Görev 3:  RF Skorunun Hesaplanması

rfm['recency_score'] = pd.qcut(rfm['recency'],5, labels=[5,4,3,2,1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'),5, labels=[1,2,3,4,5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'],5, labels=[1,2,3,4,5])
rfm['RF_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

### Görev 4:  RF Skorunun SegmentOlarak Tanımlanması
seg_map = {
    r'[1-2][1-2]' : 'hibernating',
    r'[1-2][3-4]' : 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]' : 'about_to_sleep',
    r'33' : 'need_attention',
    r'[3-4][4-5]' : 'loyal_customers',
    r'41' : 'promising',
    r'51' : 'new_customers',
    r'[4-5][2-3]' : 'potential_loyalists',
    r'5[4-5]' : 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# Görev 5:  Aksiyon Zamanı !

# Adım1:  Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz
rfm[['segment','recency','frequency','monetary']].groupby('segment').agg({'recency':'mean',
                                                                        'frequency':'mean',
                                                                        'monetary':'mean',})


# Adım2:  RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#  tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#  iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#  yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

new_df = pd.DataFrame()
rfm[(rfm['segment'] == 'champions') | (rfm['segment'] == 'loyal_customers')].index
new_df['master_id'] = df[df['interested_in_categories_12'] == '[KADIN]']['master_id']
new_df = pd.merge(new_df,rfm, how='left', on='master_id')
rfm_final = pd.DataFrame()
rfm_final = new_df[(new_df['segment'] == 'champions') | (new_df['segment'] == 'loyal_customers')]

df[df['master_id'] == 'cc294636-19f0-11eb-8d74-000d3a38a36f']
rfm.reset_index(inplace=True)
rfm[rfm['master_id']== 'cc294636-19f0-11eb-8d74-000d3a38a36f']

rfm_final.sort_values('RF_SCORE', ascending=False, inplace=True)
rfm_final.to_csv('rfm_final.csv', index=False)

# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz

rfm['segment'].value_counts()
last_df = pd.DataFrame()

rfm[(rfm['segment'] == 'about_to_sleep') | (rfm['segment'] == 'cant_loose') | (rfm['segment'] == 'new_customers')].sort_values('recency', ascending=True)


















