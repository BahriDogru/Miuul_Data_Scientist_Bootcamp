# GÖREV 1

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
d = {" Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}
t = ("Machine Learning", "Data Science")
s = ("Python", "Machine Learning", "Data Science")

print(f"x değişkeninin tipi : {type(x)}")
print(f"y değişkeninin tipi : {type(y)}")
print(f"z değişkeninin tipi : {type(z)}")
print(f"a değişkeninin tipi : {type(a)}")
print(f"b değişkeninin tipi : {type(b)}")
print(f"c değişkeninin tipi : {type(c)}")
print(f"d değişkeninin tipi : {type(d)}")
print(f"t değişkeninin tipi : {type(t)}")
print(f"s değişkeninin tipi : {type(s)}")


# GÖREV 2

def convertor(string):
    words = []
    string = string.upper()
    string = string.replace(","," ").replace("."," ")
    words = string.split()
    print(words)

text= "The goal is to turn data into information, and Information into insight. "
convertor(text)


# GÖREV 3

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

#Adım1: Verilen listenin eleman sayısına bakınız.
print("Listenin eleman sayısı:", len(lst))
#Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print("listenin sıfıncı indeksindeki eleman:", lst[0])
print("listenin onuncu indeksindeki eleman:", lst[10])
#Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
new_list = lst[0:4]
print("new_list:",new_list)
#Adım4: Sekizinci indeksteki elemanı siliniz.
lst.remove(lst[8])
print("Sekizinci indeksteki eleman silime:", lst)
#Adım5: Yeni bir eleman ekleyiniz.
lst.append("Miuul")
print("yeni eleman ekleme", lst)
#Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8,"N")
print("8. indekse 'N' değerini ekleme", lst)


# Görev 4

dict = { 'Christian' : ["America" ,18],
         'Daisy' : ["England", 12],
         'Antonio' : ["Spain", 22],
         'Dante' : ["Italy", 25]
         }

#Adım 1: Key değerlerine erişiniz.
print("Key değerleri:", dict.keys())
#Adım 2: Value'lara erişiniz.
print("Value değerleri:", dict.values())
#Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13 # dict.update({"Daisy" : ["England", 13]})
print("Daisy değiştirilen eleman:",dict["Daisy"])
#Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan geni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
#Adım 5: Antonio'yu dictionary'den siliniz.
del dict["Antonio"]
print(dict)



# Görev 5

l = [2, 13,18,93,22]

def function(list):
    o = []
    e = []
    for item in list:
        if item % 2 ==0:
            o.append(item)
        else:
            e.append(item)
    return o, e

odd , even = function(l)

print("Tek sayılar:", odd)
print("Çift sayılar:", even)


# Görev 6

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
tip_index = 1

for index, ogrenci in enumerate(ogrenciler):
    if index <=2:
        print(f"Mühendislik Fakültesi {index +1}. öğrenci :{ogrenci}")
    else:
        print(f"Tıp Fakültesi {tip_index}. öğrenci :{ogrenci}")
        tip_index +=1

# Görev 7

ders_kodu = [ "CMP1005" , "PSY1OO1" , "HUK1005" , "SEN2204" ]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

list_1 = list(zip(kredi,ders_kodu,kontenjan))

for i in list_1:
    print(f"Kredisi {i[0]} olan {i[1]} kodlu dersin kontenjanı {i[2]} kişidir.")

# Görev 8

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul" ])

if kume1.issuperset(kume2):
    print("Ortak elemanlar:",kume1  & kume2)
else:
    print("2. küümenin 1. kümeden farkı:",kume2 - kume1)