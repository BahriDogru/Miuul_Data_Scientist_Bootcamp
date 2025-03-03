# İş Problemi
# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir.
# Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız.
# Ürün önerileri 1 tane ya da l'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri
# üzerinden türetiniz.
# Kullanıcı 1'in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2'in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3'in sepetinde bulunan ürünün id'si : 22747


# Veri Seti Hikayesi
# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış
# işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi
# mevcuttur.

###########################
# Görev 1: Veriyi Hazırlama
###########################
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

#  Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel('datasets/online_retail_II.xlsx',
                    sheet_name='Year 2010-2011')
df = df_.copy()

#  Adım 2: StockCode’u POST olan  gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.
df = df[df['StockCode'] !='POST']
#df.drop(df[df['StockCode'] !='POST'].index)

# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.isnull().sum()
df = df.dropna()

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df['Invoice'].str.contains('C', na=False)]

#  Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz
df.describe().T
df = df[df['Price'] > 0]


#  Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız
df.describe().T
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return up_limit, low_limit

def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_threshold(df, 'Price')
replace_with_threshold(df, 'Quantity')


##################################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
##################################################################

# Adım 1:Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız
# df_gr = df[df['Country'] == 'Germany']
# df_gr.groupby(['Invoice','Description']).agg({'Quantity':'sum'}).unstack().iloc[0:5,0:5]
# df_gr.groupby(['Invoice','Description']).agg({'Quantity':'sum'}).unstack().fillna(0).iloc[0:5,0:5]

def create_invoice_product_df(dataframe, id=True):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum(). \
            unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)

    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum(). \
            unstack().fillna(0).map(lambda x : 1 if x >0 else 0)


# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz

def create_rules(dataframe, id=True, country='Germany'):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe_pivot = create_invoice_product_df(dataframe, id)
    frequents_itemsets = apriori(dataframe_pivot,min_support=0.01, use_colnames=True)
    rules = association_rules(frequents_itemsets, metric='support', min_threshold=0.02)
    return rules

rules = create_rules(df, id=True, country='Germany')


########################################################################################
# Görev 2: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
########################################################################################

# Adım 1:check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz

def check_id(stock_code):
    return df[df['StockCode'] == stock_code]['Description'].values[0]

#  Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz

def arl_recommender(product_id, size=5):
    recommender_list = []
    sorted_rules = rules.sort_values(by='lift',ascending=False)
    for i, product in enumerate(sorted_rules['antecedents']):
        for j in list(product):
            recommender_list.extend(list(sorted_rules.iloc[i]['consequents']))
    return recommender_list[:size]



product_id = df['StockCode'].sample(n=1).values[0]
recommender_list = arl_recommender(product_id)

#  Adım 3: Önerilecek ürünlerin isimlerine bakınız.
for item in recommender_list:
    print(check_id(item))





















