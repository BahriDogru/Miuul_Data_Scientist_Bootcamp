##########################################
# İş Problemi
##########################################


#  Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi beklenmektedir.


##########################################
# Veri Seti Hikayesi
##########################################

#  Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
#  bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
#  gösterir.


#  CustomerId : Müşteri İd’si
#  Gender : Cinsiyet
#  SeniorCitizen : Müşterinin yaşlı olup olmadığı(1, 0)
#  Partner : Müşterinin bir ortağı olup olmadığı(Evet, Hayır)
#  Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır)
#  tenure : Müşterinin şirkette kaldığı ay sayısı
#  PhoneService : Müşterinin telefon hizmeti olup olmadığı(Evet, Hayır)
#  MultipleLines : Müşterinin birden fazla hattı olup olmadığı(Evet, Hayır, Telefon hizmeti yok)
#  InternetService : Müşterinin internet servis sağlayıcısı(DSL, Fiber optik, Hayır)
#  OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
#  OnlineBackup : Müşterinin online yedeğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
#  DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
#  TechSupport : Müşterinin teknik destek alıp almadığı(Evet, Hayır, İnternet hizmeti yok)
#  StreamingTV : Müşterinin TV yayını olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
#  StreamingMovies : Müşterinin film akışı olup olmadığı(Evet, Hayır, İnternet hizmeti yok)
#  Contract : Müşterinin sözleşme süresi(Aydan aya, Bir yıl, İkiyıl)
#  PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı(Evet, Hayır)
#  PaymentMethod : Müşterinin ödeme yöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik), Kredikartı(otomatik))
#  MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
#  TotalCharges : Müşteriden tahsil edilen toplam tutar
#  Churn : Müşterinin kullanıp kullanmadığı(Evet veyaHayır)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

##########################################
# Görev 1 : Keşifçi Veri Analizi
##########################################
df.head()
df.shape
df.info()
df.isna().sum()

#  Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df["TotalCharges"] = df["TotalCharges"].replace(" ",None)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#  Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz
df.describe().T
df["InternetService"].value_counts()
df.groupby(["Churn","gender"]).agg({"tenure": "mean",
                          "MonthlyCharges" : "mean",
                          "TotalCharges" : "sum"})

df.groupby(["InternetService","gender"]).agg({"InternetService": "count",
                          "MonthlyCharges" : "mean",
                          "TotalCharges" : "sum"})


df.groupby("SeniorCitizen").agg({"gender":"count"})
df.groupby("PaymentMethod").agg({"MonthlyCharges": ["sum","count"]})


#  Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız

# Contract ve Churn ilişkisini görmek için çapraz tablo
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize='index') * 100
print(contract_churn)


fig, axes = plt.subplots(3,2, figsize=(16,10))

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0,0])
axes[0,0].set_title("Contract Türüne Göre Churn Dağılımı")


sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[0,1])
axes[0,1].set_title("PaymentMethod Türüne Göre Churn Dağılımı")


sns.countplot(data=df, x="TechSupport", hue="Churn", ax=axes[1,0])
axes[1,0].set_title("TechSupport'e Göre Churn Dağılımı")

sns.countplot(data=df, x="OnlineSecurity", hue="Churn", ax=axes[1,1])
axes[1,1].set_title("OnlineSecurity'e Göre Churn Dağılımı")

sns.countplot(data=df, x="SeniorCitizen", hue="Churn", ax=axes[2,0])
axes[2,0].set_title("SeniorCitizen'e Göre Churn Dağılımı")

plt.tight_layout()
plt.show()


#  Adım 5: Aykırı gözlem var mı inceleyiniz.

fig2, axes2 = plt.subplots(2,2, figsize=(20,14))

sns.boxplot(data=df, x="TotalCharges", ax=axes2[0,0])
axes2[0,0].set_title("TotalCharges Outliers")

sns.boxplot(data=df, x="tenure", ax=axes2[0,1])
axes2[0,1].set_title("tenure Outliers")

sns.boxplot(data=df, x="MonthlyCharges", ax=axes2[1,0])
axes2[1,0].set_title("MonthlyCharges Outliers")


plt.tight_layout()
plt.show()



#  Adım 6: Eksik gözlem var mı inceleyiniz
df.isna().sum()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())


##########################################
# Görev 2 : Feature Engineering
##########################################

# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız


# Adım 2: Yeni değişkenler oluşturunuz.


# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
def label_encoder(dataframe, column):
    labelencoder = LabelEncoder()
    dataframe[column] = labelencoder.fit_transform(dataframe[column])
    return dataframe

def one_hot_encoder(dataframe, columns):
    dataframe = pd.get_dummies(data=dataframe, columns=columns, drop_first=True)
    return dataframe

columns_for_label = [col for col in df.columns if df[col].dtypes not in [int, float] and df[col].nunique() == 2]
for col in columns_for_label:
    label_encoder(df,col)

df.head()
df.info()
columns_for_onehot = [col for col in df.columns if 2 < df[col].nunique() < 10]

df = one_hot_encoder(df, columns_for_onehot)

df.head()



#  Adım 4: Numerik değişkenler için standartlaştırma yapınız.

numeric_columns = ["tenure","TotalCharges","MonthlyCharges"]
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df.head()
df.info()


##########################################
# Görev 3 : Modelleme
##########################################


### Logistic Regression

X = df.drop(["customerID", "Churn"], axis=1)
y = df[["Churn"]]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=17)

logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)

# sabit
logistic_model.intercept_[0] # -1.2416284

# Katsayılar
logistic_model.coef_[0]


y_predict = logistic_model.predict(X_test)[:5]
y_prob = logistic_model.predict_proba(X_test)[:5]

# Model başarısı değerlendirme

print(classification_report(y_test, y_predict))
# Accuracy: 0.80
# Precision: 0.67
# Recall: 0.53
# F1-score: 0.59

# cross validation
logistic_model_for_CV = LogisticRegression()

cv_scores = cross_validate(estimator=logistic_model_for_CV,
                           X=X_train,
                           y=y_train,
                           cv=5,
                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_scores['test_accuracy'].mean()
# Accuracy: 0.804

cv_scores['test_precision'].mean()
# Precision: 0.656

cv_scores['test_recall'].mean()
# Recall: 0.551

cv_scores['test_f1'].mean()
# F1-score: 0.598

cv_scores['test_roc_auc'].mean()
# AUC: 0.848

## GridSearchCV ile en iyi hiperparametreleri bulma
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(estimator=logistic_model_for_CV,
                           param_grid=param_grid,
                           cv=5,
                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                           refit="f1")
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)

## En iyi parametreler ile modeli tekrardan eğitme
final_logistic_model = LogisticRegression(C=100)
final_logistic_model.fit(X_train, y_train)

y_pred_ = final_logistic_model.predict(X_test)

print(classification_report(y_test,y_pred_))
# Accuracy: 0.80
# Precision: 0.68
# Recall: 0.52
# F1-score: 0.59


#### KNN

knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print(classification_report(y_test, y_pred_knn))
# Accuracy: 0.75
# Precision: 0.55
# Recall: 0.50
# F1-score: 0.52


## Hiperparametre optimizasyonu
knn_model_param = KNeighborsClassifier()

knn_params = {"n_neighbors" : range(2,50)}

knn_gs_best = GridSearchCV(estimator=knn_model_param, param_grid=knn_params,cv=5, n_jobs=-1, verbose=1).fit(X_train,y_train)

knn_gs_best.best_params_

# en iyi parametreye göre modeli yeniden kuralım.
knn_final = KNeighborsClassifier(n_neighbors=36).fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

print(classification_report(y_test, y_pred_final))
# Accuracy: 0.80
# Precision: 0.65
# Recall: 0.53
# F1-score: 0.58




