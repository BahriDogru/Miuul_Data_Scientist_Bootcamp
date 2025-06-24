
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayar veya akıllı telefon üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

# Veri Seti

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

############################
#  Görev 1: Veriyi Hazırlama
############################
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)


# Adım 1:  armut_data.csv dosyasını okutunuz.
df_ = pd.read_csv('datasets/armut_data.csv')
df = df_.copy()

df.head()
df.info()
df.isnull().sum()

# Adım 2:
# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_"  ile birleştirerek
# bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()


# Adım 3:
# Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır,
# herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4,  46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı 9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek SepetId adında yeni bir değişkene atayınız.
df[df['UserId'] == 7256]

df['CreateDate'] = pd.to_datetime(df['CreateDate'])



# %Y: Dört haneli yıl (örn: 2025)
# %m: Sıfır dolgulu ay numarası (01-12) (örn: 06)
# %d: Sıfır dolgulu gün numarası (01-31) (örn: 24)
# %H: 24 saat formatında saat (00-23) (örn: 17)
# %I: 12 saat formatında saat (01-12) (örn: 05)
# %M: Sıfır dolgulu dakika (00-59) (örn: 22)

df['NEW_date'] = df['CreateDate'].dt.strftime('%Y-%m')
df.head()

df['SepetID'] = df['UserId'].astype(str) +'_'+ df['NEW_date']
#df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

###############################################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
###############################################################

# Adım 1:Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..



invoice_product_df = df.groupby(["SepetID","Hizmet"])["Hizmet"].count().\
     unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()


df["exists"] = 1 # SepetId içerisinde hizmet var mı yok mu? temsil etmek için kullandık.
df_pivot = df.pivot_table(index=['SepetID'], columns=['Hizmet'], values='exists', fill_value=0).astype(int)
# invoice_product_df = invoice_product_df.applymap(lambda x: 1 if x > 0 else 0)

# Adım 2:  Birliktelik kurallarını oluşturunuz.
# apriori fonksiyonu yardımı ile olası tüm hizmetlerin birlikteliklerinin support değerlerini çıkartacağız

frequent_services = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
frequent_services.sort_values(by='support', ascending=False)


rules = association_rules(frequent_services,
                          metric='support',
                          min_threshold=0.01)


# Adım3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

hizmet= '22_0'

def arl_recommender(rules, hizmet, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # ürün çiftlerinden kurtulmak için
            if j == hizmet:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


recommendation_list = arl_recommender(rules, hizmet, rec_count=3)

########################################################################################

def r_c_s(sorted_rules,main_service_,n=0):
    recommendation_list = []
    l = []
    for i,product in enumerate(sorted_rules["antecedents"]):
        if len(product) == 1:
            for j in list(product):
                if (j == main_service_):
                    l = []
                    for i,co_i_p in enumerate(sorted_rules.iloc[i]["consequents"]):
                        l.append(co_i_p)
        recommendation_list.append(l)


    recommendation_set = set(map(tuple, recommendation_list))
    recommendation_list = list(recommendation_set)
    print(recommendation_list[0:n])
    return recommendation_list[0:n]

sorted_rules = rules.sort_values("lift", ascending=False)


main_service = "2_0"
recomend_main_service = r_c_s(sorted_rules,main_service,5)
