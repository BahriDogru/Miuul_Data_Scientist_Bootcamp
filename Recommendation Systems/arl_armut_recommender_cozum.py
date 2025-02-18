
############################
#  Görev 1: Veriyi Hazırlama
############################
import pandas as pd
import warnings
from warnings import filterwarnings
filterwarnings('ignore')
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)


#Adım 1:  armut_data.csv dosyasını okutunuz.
df_ = pd.read_csv('datasets/armut_data.csv')
df = df_.copy()

df.info()
df.isnull().sum()



# Adım 2:
# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.  ServiceID ve CategoryID’yi "_"  ile birleştirerek bu hizmetleri
#  temsil edecek yeni bir değişken oluşturunuz.


df['hizmet'] = df['ServiceId'].astype(str)+ '_' + df['CategoryId'].astype(str)


# Adım 3:
# Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı
#  hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4,  46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4  hizmetleri
#  başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
#  date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek SepetId adında yeni bir değişkene atayınız.
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['new_date'] = df['CreateDate'].dt.strftime('%Y-%m')

df['SepetId'] = df['UserId'].astype(str) +'_'+ df['new_date'].astype(str)


###############################################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
###############################################################

df['UserId'].nunique()
df['SepetId'].nunique()
df['hizmet'].nunique()
df['hizmet'].value_counts() # Veri setinde küçükltme gerekip gerekmediğine bakmak istedim.


# Adım 1:Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

df["exists"] = 1 # SepetId içerisinde hizmet var mı yok mu? temsil etmek için kullandık.

basket_service_df = df.pivot_table(index=['SepetId'], columns=['hizmet'], values='exists', fill_value=0)

# df.pivot_table(index = "SepetId", columns="Hizmet", values = "UserId").notna().astype(int)
#
#
# df_pivot = df.groupby(["UserId","Service"]).agg({"ServiceId" : "sum"}).\
#     unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


basket_service_df = basket_service_df.applymap(lambda x: 0 if x < 1 else 1)
basket_service_df = basket_service_df.astype(bool)
# Adım 2:  Birliktelik kurallarını oluşturunuz.
# apriori fonksiyonu yardımı ile olası tüm hizmetlerin birlikteliklerinin support değerlerini çıkartacağız

frequent_services = apriori(basket_service_df,min_support=0.005, use_colnames=True)
frequent_services.sort_values(by='support', ascending=True)

rules = association_rules(frequent_services,
                          metric='support',
                          min_threshold=0.001)


rules[(rules['support'] > 0.005) & (rules['confidence'] > 0.01) & (rules['lift'] > 5)]

#  Adım3:  arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

hizmet= '2_0'

def arl_recommender(rules, hizmet, rec_count=5):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        if hizmet in product:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[:rec_count]


recommendation_list = arl_recommender(rules, hizmet, rec_count=5)

############################################################################################

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









