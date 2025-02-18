# İş Problemi
# ID'sİ verilen kullanıcı İçin İtem-based ve user-based recommender
# yöntemlerini kullanarak 10 film önerisi yapınız.


# Veri Seti Hikayesi
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan
# derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme İçermektedir. Bu veri seti İse 17 Ekim 2016
# tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 İle 31 Mart 2015 tarihleri arasında verileri İçermektedir. Kullanıcılar
# rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

import pandas as pd
import warnings
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

############################
# User Based Recommendation
############################

### Görev 1:  Veri Hazırlama

#Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# Adım 2:  rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.

df = movie.merge(rating, how='left', on="movieId")
df = df[:2000000]
# Adım 3:  Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.

total_rating_count = pd.DataFrame(df['title'].value_counts())
# total_rating_count.reset_index(inplace=True)
# total_rating_count = total_rating_count[total_rating_count['count']>1000]
rare_movies = total_rating_count[total_rating_count['count'] < 1000].index


#  Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz
common_movies = df[~df['title'].isin(rare_movies)]
movie_user_df = common_movies.pivot_table(values='rating',index=['userId'],columns=['title'])

# Adım5:  Yapılan tüm işlemleri fonksiyonlaştırınız.

def create_movie_user_df(obs=2000000):
    import pandas as pd
    movie = pd.read_csv('datas/movie.csv')
    rating = pd.read_csv('datas/rating.csv')
    df = movie.merge(rating, how='left', on="movieId")
    df = df[:obs]
    total_rating_count = pd.DataFrame(df['title'].value_counts())
    rare_movies = total_rating_count[total_rating_count['count'] < 1000].index
    common_movies = df[~df['title'].isin(rare_movies)]
    movie_user_df = common_movies.pivot_table(values='rating', index=['userId'], columns=['title'])
    return movie_user_df


### Görev 2:  Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi

#  Adım 1: Rastgele bir kullanıcı id’si seçiniz.
random_user = int(df['userId'].sample(n=1,random_state=45).values[0])

# Adım 2:  Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = movie_user_df[movie_user_df.index==random_user]

#  Adım3:  Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()


###  Görev 3:  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi

# Adım 1:  Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_dfadında yeni bir dataframe oluşturunuz.
movies_watched_df = movie_user_df[movies_watched]

# Adım 2:  Her birkullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe
#  oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ['userId','movie_count']
# Adım3:  Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste
#  oluşturunuz.

users_same_movies = user_movie_count[user_movie_count['movie_count'] >= (len(movies_watched) * 60/100)]['userId']

###  Görev 4:  Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi

#  Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df
#  dataframe’ini filtreleyiniz

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

#  Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=['Corr'])
corr_df.index.names = ['userId1','userId2']
corr_df.reset_index(inplace=True)

# Adım3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe
#  oluşturunuz

top_users = corr_df[(corr_df['userId1']==random_user) & (corr_df['Corr'] > 0.65)][['userId2','Corr']].reset_index(drop=True) # Burada corrdf içerisinden gelen indexi atıyor.
top_users.sort_values(by='Corr', ascending=False, inplace=True)
top_users.columns = ['userId', 'Corr']

#  Adım4:  top_users dataframe’ine rating veri seti ile merge ediniz.

top_users_rating = top_users.merge(rating[['userId', 'movieId', 'rating']],how='inner', on="userId")



### Görev 5:  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması

# Adım 1:   Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_ratingadında yeni bir değişken oluşturunuz.

top_users_rating['weighted_rating'] = top_users_rating['Corr'] * top_users_rating['rating']

#  Adım 2:  Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_dfadındayeni bir
#  dataframe oluşturunuz.

recommendation_df = top_users_rating.groupby('movieId').agg({'weighted_rating': 'mean'})
recommendation_df.reset_index(inplace=True)


#  Adım3:  recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız

recommendation_df[recommendation_df['weighted_rating'] > 3.5].sort_values(by='weighted_rating', ascending=False)
movies_to_be_recommend = recommendation_df[recommendation_df['weighted_rating'] > 3.5].sort_values(by='weighted_rating', ascending=False)


#  Adım4:  movie verisetinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz
def movie_recommendation(mtbr,size=5):
    movies_to_be_recommend_list = mtbr.merge(movie[['movieId', 'title']], on='movieId')['title']
    return movies_to_be_recommend_list[:size]

list = movie_recommendation(movies_to_be_recommend,size=5)




############################
# Item Based Recommendation
############################

# Görev 1:  Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.

# Adım 1:   movie, rating veri setlerini okutunuz.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')
df = movie.merge(rating, how='left', on="movieId")
df = df[:2000000]

# Adım 2:  Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
df['timestamp'] = pd.to_datetime(df['timestamp'])

last_movies_id = (df[(df['rating'] == 5.0) & (df['userId']== random_user)]. \
               sort_values(by='timestamp', ascending=False)['movieId'].tolist())[0]


# Adım3:  User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz
movie_name = df[df['movieId']==last_movies_id].iloc[0]['title']
# movie_name = pd.Series(movie_user_df.columns).sample(1).values[0]
filtered_df = movie_user_df[movie_name]

# Adım 4:  Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız
movie_user_df.corrwith(filtered_df).sort_values(ascending=False).head(10)


#  Adım5:  Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movie_list = (movie_user_df.corrwith(filtered_df).sort_values(ascending=False).index[1:6]).tolist()
movie_list













