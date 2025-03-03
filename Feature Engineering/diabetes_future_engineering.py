import numpy as np
import pandas as pd
import seaborn as sns
from click import style
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


#####################
# İş Problemi
#####################
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir



#####################
# Veri Seti Hikayesi
#####################
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir



########################################
# Görev 1 : Keşifçi Veri Analizi
########################################

def load_dataset():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

# Adım 1: Genel resmi inceleyiniz
df = load_dataset()
df.shape
df.head()
df.info()
df.isnull().sum()
df.describe().T

#  Adım 2: Numerik ve kategorik değişkenleri yakalayınız
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

# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız
df['Pregnancies'].value_counts()
df.describe().T
df.groupby('Outcome').agg({"Age": "mean",
                           "BMI":"mean",
                           "Outcome": "count",
                           "Pregnancies": "mean"})


#  Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

df.groupby('Outcome').agg({"Age": "mean",
                           "BMI":"mean",
                           "Outcome": "count",
                           "Pregnancies": "mean"})


#  Adım 6: Eksik gözlem analizi yapınız.
df.isnull().sum()


# Adım 7: Korelasyon analizi yapınız
df.plot.scatter(x="Outcome", y="BMI")
plt.show()

df.plot.scatter(x="Pregnancies", y="BMI")
plt.show()

msno.heatmap(df)
plt.show()

################################################
# Görev 2 : Feature Engineering
################################################

#####################################################################################
# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df.head()
df[["Glucose","Insulin"]].describe([0.01, 0.25, 0.5, 0.75, 0.95, 0.99 ,1]).T
len(df[df["Glucose"] == 0].index) # 5 gözlem biriminde Glucose değeri 0
len(df[df["Insulin"] == 0].index) # 374 gözlem biriminde Insulin değeri 0
len(df[df["BMI"] == 0 ].index) # 11 gözlem biriminde BMI değeri 0
df.loc[df["Glucose"] == 0, "Glucose"] = None
df.loc[df["Insulin"] == 0, "Insulin"] = None
df.loc[df["BMI"] == 0, "BMI"] = None

def zero_to_None(dataframe,col):
    dataframe.loc[dataframe[col] == 0, col] = None
cols = ["BloodPressure", "SkinThickness", "Age", "Glucose", "Insulin" ,"BMI"]

df = load_dataset()

for col in cols:
    zero_to_None(df,col)

df.isnull().sum()
df.head()


##### Outlier değerlerin giderilmesi #####

for col in df.columns:
    sns.boxplot(x=df[col])
    plt.show()

def outliers_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    inter_quartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * inter_quartile_range
    low_limit = quartile1 - 1.5 * inter_quartile_range
    return low_limit, up_limit
def check_outliers(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# Outlier olan ve olmayan değişkenler
for col in df.columns:
    print(col, check_outliers(df, col))

df.describe([0.05, 0.25, 0.5, 0.75, 0.95]).T

# Outlier olan değerleri threshold değeri ile değiştirmek
for col in df.columns:
    replace_with_thresholds(df,col)

df.describe([0.05, 0.25, 0.5, 0.75, 0.95]).T


##### Eksik değerlerin giderilmesi #####
df["Glucose"].mean() # 121.6867627785059
df["BloodPressure"].mean() # 72.40518417462484
df["SkinThickness"].mean() # 29.153419593345657
df["Insulin"].mean() # 155.5482233502538
df["BMI"].mean() # 32.457463672391015


# Burada eksik değerleri Tahmine dayalı yöntemler ile gidereceğiz.
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()

# Şimdi model kurarak eksik değrleri doldralım.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()

# Scaler işlemini geri alma
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()

################################################################################
###  Adım 2: Yeni değişkenler oluşturunuz

df.loc[(df["Age"] < 30), "Age_Cat"] = "young"
df.loc[(df["Age"] >= 30) & (df["Age"] < 50) , "Age_Cat"] = "adult"
df.loc[(df["Age"] >= 50), "Age_Cat"] = "old"

df.loc[df["BMI"] < 18.5, "BMI_Cat"] = "underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 25), "BMI_Cat"] = "normal"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), "BMI_Cat"] = "overweight"
df.loc[(df["BMI"] >= 30) & (df["BMI"] < 35), "BMI_Cat"] = "obese"
df.loc[df["BMI"] >= 35, "BMI_Cat"] = "extremely_obese"


df.head()


####################################################################################
#  Adım 3:  Encoding işlemlerini gerçekleştiriniz
df["Age_Cat"].value_counts()
df["BMI_Cat"].value_counts()

df = pd.get_dummies(df, columns=["Age_Cat","BMI_Cat"], drop_first=True)


####################################################################################
#  Adım 4: Numerik değişkenler için standartlaştırma yapınız.
df.info()

mms = MinMaxScaler()
df = pd.DataFrame(mms.fit_transform(df), columns=df.columns)


####################################################################################
# Adım 5: Model oluşturunuz.

y = df["Outcome"]
X = df.drop( "Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)













