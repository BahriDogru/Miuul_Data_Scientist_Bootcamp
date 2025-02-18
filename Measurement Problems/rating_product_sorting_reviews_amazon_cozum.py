import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('datasets/amazon_review.csv')
drop_list = ['reviewerName','reviewText','summary','unixReviewTime']
df.drop(drop_list, axis=1, inplace=True)
df.head()
df.shape
df.isnull().sum()

##### Görev 1 ########
# Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen puanları tarihe göre
# ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir

# Adım 1:   Ürünün ortalama puanını hesaplayınız.
df['overall'].head(20)
df['overall'].mean() #4.587589013224822

#  Adım 2:  Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
df.info()
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df['reviewTime'].max()

#son 30 gün puan ortalaması
df.loc[df['day_diff'] < 30, 'overall'].mean()
# df.loc[df['day_diff'] <= df['day_diff'].quantile(0.25), 'overall'].mean()

df.loc[df['day_diff'] <= 30, 'overall'].mean() #4.742424242424242
df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean()  #4.803149606299213
df.loc[(df['day_diff'] < 90) & (df['day_diff'] <= 180), 'overall'].mean() #4.782383419689119
df.loc[df['day_diff'] > 180, 'overall'].mean() #4.573373327180434




def time_based_weighted_average(dataframe, w1=28,w2=26,w3=24,w4=22):
    return dataframe.loc[dataframe['day_diff'] <= 30, 'overall'].mean() * w1/100 + \
        dataframe.loc[(dataframe['day_diff'] > 30) & (dataframe['day_diff'] <= 90), 'overall'].mean() * w2/100 + \
        dataframe.loc[(dataframe['day_diff'] < 90) & (dataframe['day_diff'] <= 180), 'overall'].mean() * w3/100 + \
        dataframe.loc[dataframe['day_diff'] > 180, 'overall'].mean() * w4/100

time_based_weighted_average(df) #4.730611838221668

# Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız

df.loc[df['day_diff'] <= 30, 'overall'].mean() #4.742424242424242
df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean()  #4.803149606299213
df.loc[(df['day_diff'] < 90) & (df['day_diff'] <= 180), 'overall'].mean() #4.782383419689119
df.loc[df['day_diff'] > 180, 'overall'].mean() #4.573373327180434



###### Görev 2 #######
# Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# Adım 1:  helpful_no değişkenini üretiniz.
df.head()
df['helpful_no'] = df['total_vote'] - df['helpful_yes']


#  Adım 2:  score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz

def score_pos_neg_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['score_pos_neg_diff'] = df.apply(lambda x:score_pos_neg_diff(x['helpful_yes'], x['helpful_no']), axis=1)
df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1)
df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)

# Adım 3:  20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
#  • wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
df.sort_values(by=['wilson_lower_bound'], ascending=False).head(20)


