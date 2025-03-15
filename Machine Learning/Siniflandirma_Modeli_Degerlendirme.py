import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

data = {
    "values" : [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "predict_values" : [ 0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]
}

df = pd.DataFrame(data)

## Görev 1:

#  Müşterinin churn olup olmama durumunu tahminleyen bir
#  sınıflandırma modeli oluşturulmuştur. 10 test verisi gözleminin
#  gerçek değerleri ve modelin tahmin ettiği olasılık değerleri
#  verilmiştir.


# - Eşik değerini 0.5 alarak confusion matrix oluşturunuz.
# - Accuracy,Recall, Precision, F1 Skorlarını hesaplayınız

df["y_predict"] = [1 if value > 0.50 else 0 for value in df["model_predict_values"].values]

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(df["values"], df["y_predict"])
confusion_matrix(df["values"], df["y_predict"])
                                     # Model tahmini
#                                  Churn(1)     Non-churn(0)

# Gerçek        Churn(1)               4              2
# değerler      Non-churn(0)           1              3

print(classification_report(df["values"], df["y_predict"]))
# accuracy : 0.70 (TP + TN / All)
# precision : 0.80 (TP / TP + FP)
# recall : 0.67  (TP / TP + FN)
# f1-score : 0.73 (2 * (Precision * Recall) / (Precision + Recall))


## Görev 2:

# Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur. %90.5 doğruluk
#  oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır. Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi
#  olmamış, iş birimi modelin başarısız olduğunu iletmiştir. Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir. Buna göre;

                                     # Model tahmini
#                                  Fraud(1)     Non-Fraud(0)

# Gerçek        Fraud(1)               5              5
# değerler      Non-Fraud(0)           90             900



# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.
# accuracy : 0.90 (TP + TN / All)
# precision : 0.05 (TP / TP + FP)
# recall : 0.50  (TP / TP + FN)
# f1-score : 0.09 (2 * (Precision * Recall) / (Precision + Recall))


# - Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.

# Veri seti dengeli sınıf dağılımına sahip değil
# Accuracy değerine göre sadece değerlendirme yapılması yanlıştır.
# Bu tarz sınıf dağılımı eşit olmayan veri setlerinde precision ve recall dğerlerine de ayrıca bakılmalıdır.
# Tip 2 hatası !!!





