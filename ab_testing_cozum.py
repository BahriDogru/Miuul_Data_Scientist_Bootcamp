import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import  proportions_ztest
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.


#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################
# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control_group = pd.read_excel('datas/ab_testing.xlsx', sheet_name='Control Group')
test_group = pd.read_excel('datas/ab_testing.xlsx', sheet_name='Test Group')

#  Adım 2: Kontrol ve test grubu verilerini analiz ediniz
test_group.describe().T
control_group.describe().T

#  Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat([control_group, test_group], ignore_index=True)



#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0: M1 = M2 ( Test ve Kontrol grupları arasında "Purchase" açısından istatistiksel olarak anlamlı bir fark YOKTUR)
# H1: M1 != M2 ( Test ve Kontrol grupları arasında "Purchase" açısından istatistiksel olarak anlamlı bir fark VARDIR)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
control_group['Purchase'].mean() # 550.8940587702316
test_group['Purchase'].mean() # 550.8940587702316


import scipy.stats as stats
import seaborn as sns

plt.subplot(1, 2, 2)
stats.probplot(control_group["Purchase"], dist="norm", plot=plt)
plt.title("QQ Plot - Control Group")
plt.show()

plt.subplot(1, 2, 1)
sns.histplot(control_group["Purchase"], kde=True, color="blue", bins=15, label="Control Group")
sns.histplot(test_group["Purchase"], kde=True, color="red", bins=15, label="Test Group")
plt.legend()
plt.title("Count - Purchase Dağılımı")
plt.show()


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
    # Normallik Varsayımı :
        # H0: Normal dağılım varsayımı sağlanmaktadır.
        # H1: Normal dağılım varsayımı sağlanmamaktadır.
        # p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
        # Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.

test_stat, p_value = shapiro(control_group['Purchase'])
print(p_value) # 0.5891071186294093

test_stat, p_value = shapiro(test_group['Purchase'])
print(p_value) # 0.15413405050730578

# H0 Reddedilemez, Normal Dağılım Varyansı sağlanmaktadır.


    #VaryansHomojenliği :
        # H0: Varyanslar homojendir.
        # H1: Varyanslar homojen Değildir.
        # p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
        # Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
        # Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.
test_stat, p_value = levene(control_group['Purchase'], test_group['Purchase'])
print(p_value) # 0.10828588271874791
# H0 REDDEDİLEMEZ, Varyanslar homojendir.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
# Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)

# H0: M1 = M2 (İki grup arasında getiri farkı YOKTUR)
# H1: M1 != M2 (İki grup arasında getiri farkı VARDIR)

test_stat , p_value = ttest_ind(control_group['Purchase'], test_group['Purchase'])
print(p_value)  # 0.34932579202108416
# H0 REDDEDİLEMEZ,  Test ve Kontrol grupları arasında "Purchase" açısından istatistiksel olarak anlamlı bir fark yoktur.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki
# olarak anlamlı bir fark olup olmadığını yorumlayınız.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# Test ve Kontrol grupları için öncelikle 'shapiro' testini kullanıldı. Her iki gruptaki 'Purchase' değişkeninin Normal dağılıma uyup uymadığı kontrol edildi.
# Ayrıca 'Purchase' değişkeni üzerinde Varyans Homojenliği varsayımı kontolüde yapıldı.
# 'Purhase' değişkeni hem normallik hem de varyans homojenliği varsayımlarını sağladığı için Bağımsız İki Örneklem T Testi kullanıldı.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# Firmanın, Average Bidding (Test Grubu) yöntemine geçmesi için istatistiksel olarak güçlü bir kanıt bulunmuyor.
# Eğer bu yöntemin daha iyi olup olmadığını kesinleştirmek isterlerse:
# Daha fazla veri toplayabilirler
# Farklı metrikleri de değerlendirebilirler (örneğin, "Earning" üzerinde de test yapılabilinir)


