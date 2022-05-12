import math
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import sympy
from sympy import *


# print("Jak sie nazywasz?")
# name = input()
# print(f"Hello, {name}.")

# string = "zmienna"
# int = 1
# float = 3.5
# print("string: {} typ: {) int: {} typ: {)float: {} type {)".format(string, type(string), int, type(int), float, type(float))

# list = ["jeden", "dwa", "trzy"]
# y = '#'.join(list)
# print(y)
# z = y.split("#")
# print(z)


# znaki = "Metody Inżynierii Wiedzy Są Najlepsze"
# print("{} Długość:{}".format(znaki, len(znaki)))
# male = znaki.lower()
# print("{} Długość:{}".format(male, len(male)))
# bezpol = znaki.replace('ż', 'z').replace('ą', 'a').replace(' ', '')
# print("{} Długość:{}".format(bezpol, len(bezpol)))
# x = set(bezpol)
# print("{} Długość:{}".format(x, len(x)))


# string = "zmienna"
# liczba = 4
# a = (string, liczba)
# print("Typ 1:{}  Typ2:{}".format(type(a[0]), type(a[1])))

# jezyki = ['Python', 'Java', 'C++', 'C']
# jezyki2 = ['aas', 'asasas']
# print(len(jezyki))
# print(jezyki.index('Python'))
# jezyki.extend(jezyki2)
# print('{}{}'.format(jezyki, jezyki2))

# slownik = {"Niemcy": "Berlin", "Rosja": "Moskwa", "Białoruś": "Mińsk", "Litwa": "Wilno", "Ukraina": "Kijów",
#            "Słowacja": "Bratysława", "Czechy": "Praga"}
# print(sorted(slownik.values()))
# print(bool(' '))
# print(bool(''))
# print(bool('1'))
# print(bool('0'))

# zdanie ="Ala ma kota"
# if 'j' in zdanie:
#     print("Tak")
# else:
#     print("Nie")
#
# for i in range(10):
#     print(i)

# list = ["jeden", "dwa", "trzy"]
# y = '#'.join(list)
# print(y)
# tmp = ''
# for i in range(y):
#     if i != '#':
#         tmp += i
#     else:
#         print(tmp)
#

# 10 znaków
# duże małe litery
# znak specjalny

def czyMocne(haslo):
    dlugosc = len(haslo)
    flag = 0
    upper = 0
    lower = 0
    for i in haslo:
        if i == '!':
            flag = 1
        if i.isupper():
            upper = 1
        if i.islower():
            lower = 1
    if dlugosc >= 10 and flag == 1 and upper == 1 and lower == 1:
        print("haslo ok")


# czyMocne("aaaaaaaaA!aa")


# lista = [3, 2, 1, 99, 12]
# for i in list:
#     if i != 99:
#         print(i)


# 1 return czy nalezy

def czyNalezy(lista, liczba):
    while liczba in lista:
            print("Tak")
    return


# czyNalezy(lista, 12)

def czyNalezy2(lista, liczba):
    flaga = False
    i = 0
    while i < len(lista):
        if lista[i] == liczba:
            flaga = True
            break
        i += 1
    return flaga

# czyNalezy2(lista, 12)


# tworzenie piku metody inzynierii wiedzy

# f = open("text.txt", "a")
# f.write("Metody\n")
# f.write("inzynierii\n")
# f.write("wiedzy\n")
# f.close()
#
# f = open("text.txt", "r")
# for i in f:
#     print(i, end="")
# f.close()


# jezyki = ['Python', 'Java', 'C++', 'C']
#
# with open('out.txt', 'w') as f:
#     for i in jezyki:
#         print(i, file=f)


# miasta = ['Olsztyn', 'Warszawa', 'Gdańsk', 'Sopot'];
#
# # map = map(pierwsze3, miasta)
# nowa = list(map(lambda w: w[:3], miasta))
# print(nowa)

# funkcja zwracajaca nazwy plików z kropka
pliki = ['tekst.txt', 'text.doc', 'texkskks.doc', 'Sopot']


# def foo(lista, rozsz):
#     wynik = []
#     for i in lista:
#         if i.endswith(rozsz):
#             wynik.append(i)
#     return wynik


# lista = foo(pliki, ".txt")
# print(lista)


# generetor
# def foo(lista, rozsz):
#     for i in lista:
#         if i.endswith(rozsz):
#             yield i


# lista = foo(pliki,".txt")
# print(lista)


def metryka_euklidesowa(lista_a, lista_b):
    tmp = 0
    for i in range(len(lista_a)-1):
        tmp += (lista_a[i] - lista_b[i])**2
    return math.sqrt(tmp)


def zadanie1(lista):
    slownik = {}
    for i in lista[1:]:
        if i[14] not in slownik.keys():
            slownik[i[14]] = [metryka_euklidesowa(lista[0], i)]
        else:
            slownik[i[14]].append(metryka_euklidesowa(lista[0], i))
    return slownik


# print(metryka_euklidesowa(lista[0], lista[3]))
# print(zadanie1(lista)[1.0])

# m = [[1,2,3], [4,5,6], [7,8,9]]


# def wskaznik(macierz, wynik=0):
#     indeksy = list(range(len(macierz)))
#
#     if len(macierz) == 2 and len(macierz[0]) == 2:
#         wartosc = macierz[0][0] * macierz[1][1] - macierz[1][0] * macierz[0][1]
#         return wartosc
#
#     for fc in indeksy:
#         macierz_kopia = macierz.copy()
#         macierz_kopia = macierz_kopia[1:]
#         wysokosc = len(macierz_kopia)
#         for i in range(wysokosc):
#             macierz_kopia[i] = macierz_kopia[i][0:fc] + macierz_kopia[i][fc + 1:]
#
#         znak = (-1)  (fc % 2)
#         pod_wskaznik = wskaznik(macierz_kopia)
#         wynik += znak * macierz[0][fc] * pod_wskaznik
#
#     return wynik

separator = ' '
lista = []
with open("australian.dat", 'r') as file:
    for line in file:
        tmp = line.split(separator)
        tmp = list(map(lambda i: float(i), tmp))
        lista.append(tmp)
# print(lista)


def funkcja(x, lista):
    tuple = []
    for i in range(len(lista)):
        tuple.append((lista[i][len(lista[i]) - 1], metryka_euklidesowa(x, lista[i])))
    return tuple


x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# funn = funkcja([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], lista)

# print(funn)

# # dom knn co to
# # metryka w oparciu o wektor i działania na wektorze
# # funkcja biorąca lista k x   i zwroci decyzje


def odlegosciodX(lista, x):
    wynik = []
    for i in lista:
        para = (i[-1], (metryka_euklidesowa(x, i)))
        wynik.append(para)
    return wynik


def podzial(lista):
    wynik = {}
    for i in range(len(lista)):
        pom = lista[i][0]
        if pom in wynik.keys():
            wynik[pom].append(lista[i][1])
        else:
            wynik[pom] = [lista[i][1]]
        wynik[pom].sort()
    return wynik


def segregacjaOdleglosci(lista):
    slownik = {}
    for i in lista:
        if i[0] not in slownik.keys():
            slownik[i[0]] = [i[1]]
        else:
            slownik[i[0]].append(i[1])
    return slownik


def sumowanie(lista, k):
    wynik = {}
    for klucz in lista.keys():
        suma = 0
        for i in range(k):
            suma += lista[klucz][i]
        if klucz not in wynik.keys():
            wynik[klucz] = []
        wynik[klucz].append(suma)
    return wynik


def sumowanieOdleglosci(lista, k):
    slownik = {}
    for i in lista.keys():
        tmp_list = lista[i]
        tmp_list.sort()
        slownik[i] = sum(tmp_list[0:k])
    return slownik


def decyzja(lista, x, k):
    odleglosc = odlegosciodX(lista, x)
    slownik = podzial(odleglosc)
    sumaKodleglosci = sumowanie(slownik, k)
    print(sumaKodleglosci)
    list = [(k, v) for k, v in sumaKodleglosci.items()]
    min = list[0][1]
    dec = 0
    for para in list[1:]:
        if para[1] == min:
            return None
        if para[1] < min:
            min = para[1]
            dec = para[0]
    return dec


# print(decyzja(lista, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1))


#  ZAD1
# dom 28 luty godzina 10 z nagrania z wykladu
# kolorowanie na dwa kolory, najpierw każdemu wektorowi dać dowolne kolorowanie usuwajc ostatni element kazdej listy
# policzyć sumy(srodek masy) np oddzielnia funkcja takie ze wyznaczy kropke z ktorej odległośc do pozostałych jest najmniejsza i ona zostaje srodkiem masy
# nastepnie trzeba znów liczyć odległości każdej z kropek do kropkek srodka masy, jeżeli kolor np. czarny kropki do czarnej kropki jest wiekszy niz do kropki biaej zmieniamy malowanie


# ZAD2
# metoda motne carlo liczenia całki
# wyznaczamy maks z wartości funkcji x2
# kropkujemy kwadrat
# potem porównujemy wszystkie punkty z wartościa funkcji i liczymy ile nad ile pod
# potem ten współczynnik mnożymy x pole kwadratu

# ZAD3
# dzielimy funkcje na polowe, dzielimy na kwadraty i sumujemy prostokąty lub trapezy
def metryka_euklidesowa_2(lista1, lista2, utnij=False):
    if utnij:
        lista1 = lista1[:-1]
        lista2 = lista2[:-1]
    v1 = np.array(lista1)
    v2 = np.array(lista2)
    c = v1 - v2
    skalar = np.dot(c, c)
    return math.sqrt(skalar)


def kolorowanie(lista, klasyDecyzyjne):
    for wektor in lista:
        wektor[-1] = float(random.randint(0, klasyDecyzyjne-1))
    return lista


def dzielenieKolorami(lista):
    wynik = {}
    for wektor in lista:
        if wektor[-1] in wynik.keys():
            wynik[wektor[-1]].append(wektor[0:-1])
        else:
            wynik[wektor[-1]] = [wektor[0:-1]]
    return wynik


def srodekMasy(slownik):
    punktyCiezkosci = {}
    for klasa in slownik:
        minimalna = float(math.inf)
        for element in slownik[klasa]:
            sumaOdleglosci = 0
            for i in range(len(slownik[klasa])):
                sumaOdleglosci += metryka_euklidesowa_2(element, slownik[klasa][i])
            sredniaOdleglosci = sumaOdleglosci / len(slownik[klasa])
            if sredniaOdleglosci < minimalna:
                punktyCiezkosci[klasa] = (element, sredniaOdleglosci)
                minimalna = sredniaOdleglosci
    return punktyCiezkosci


def zamianaKolorowan(slownik, punktyCiezkosci):
    wynik = {}
    for klasa in slownik:
        for element in slownik[klasa]:
            minimalna = float(math.inf)
            punkt = ()

            for klasaCiezkosci in punktyCiezkosci:
                odlegloscDoPunktuCiezkosci = metryka_euklidesowa_2(element, punktyCiezkosci[klasaCiezkosci][0])
                if odlegloscDoPunktuCiezkosci < minimalna:
                    punkt = punktyCiezkosci[klasaCiezkosci]
                    minimalna = odlegloscDoPunktuCiezkosci

            for klasaCiezkosci in punktyCiezkosci:
                if punkt == punktyCiezkosci[klasaCiezkosci]:
                    if klasaCiezkosci not in wynik.keys():
                        wynik[klasaCiezkosci] = [element]
                    else:
                        wynik[klasaCiezkosci].append(element)
    return wynik


def kMeanWithoutTeacher(lista):
    slownik = dzielenieKolorami(lista)
    srodkiMasy=srodekMasy(slownik)
    zamiana = zamianaKolorowan(slownik, srodkiMasy)
    listaWynik = []
    for klasa in zamiana:
        for element in zamiana[klasa]:
            element.append(klasa)
            listaWynik.append(element)
    # for _ in range(5):
    #     return kMeanWithoutTeacher(listaWynik)
    return listaWynik



# lista_kolorowa=kolorowanie(lista, 2)
# print(dzielenieKolorami(lista_kolorowa))
# print(kMeanWithoutTeacher(lista))


def fun(x):
    return x


def calkaMonteCarlo(a, b, f, i=100):
    punkty_wewnatrz = 0
    x = np.linspace(a, b, i)
    p = []
    for k in x:
        p += [f(k)]
    y = np.linspace(min(p), 2 * max(p), i)
    liczba_punktow = len(x) * len(y)
    pole_prostokata = (abs(a) + abs(b)) * (abs(min(p)) + 2 * abs(max(p)))
    for i in range(len(x)):
        for j in range(len(y)):
            if f(x[i]) > 0 and y[j] > 0:
                if f(x[i]) >= y[j]:
                    punkty_wewnatrz += 1
            elif f(x[i]) < 0 and y[j] < 0:
                if f(x[i]) <= y[j]:
                    punkty_wewnatrz -= 1
    return pole_prostokata * (punkty_wewnatrz / liczba_punktow)


# print(calkaMonteCarlo(0, 1, fun))


def calkaProstokaty(a, b, f, i=100):
    suma = 0
    x = np.linspace(a, b, i)
    for k in range(len(x)):
        suma += f(a+k*(b - a)/i)*(b - a)/i
    return suma


# print(calkaProstokaty(0, 1, fun, 100000))


def sredniaArytmetyczna(wektor, utnij=True):
    if utnij:
        wektor=wektor[:-1]
    array = np.array(wektor)
    srednia = sum(array)/len(array)
    return srednia


def sredniaArytmetycznaWektorowo(wektor, jedynki):
    v1 = np.array(wektor)
    tmp = np.dot(v1, jedynki)
    srednia = tmp/len(v1)
    return srednia


def wariancja(wektor, utnij=True):
    srednia = sredniaArytmetyczna(wektor, utnij)
    if utnij:
        wektor=wektor[:-1]
    array = np.array(wektor)
    sum = 0
    for i in array:
        sum += (i - srednia)**2
    wariancja = sum/len(array)
    return wariancja


def wariancjaWektorowo(wektor, utnij=True):
    if utnij:
        wektor = wektor[:-1]
    srednia = sredniaArytmetycznaWektorowo(wektor, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    srednia = [srednia for _ in range(len(wektor))]
    wektor = np.subtract(wektor, srednia)
    wariancja = (np.matmul(wektor, np.transpose(wektor)))/len(wektor)
    return wariancja


def odchylenieStandardowe(wektor, utnij=True):
    wariancja1 = wariancja(wektor, utnij)
    return math.sqrt(wariancja1)


def odchylenieStandardoweWektorowo(wektor, utnij=True):
    wariancja = wariancjaWektorowo(wektor, utnij)
    return math.sqrt(wariancja)
# jezeli liczba zmian jest mniejsza niz x to zakończ ---------!!!!!!!!!!!!!!!!!! 22 min ostatni wykład



# wariancja_stara = wariancja(lista[0])
# print("Wariancja obliczona nie wektorowo: ", wariancja_stara)
# war = wariancjaWektorowo(lista[0])
# print("Wariancja obliczona wektorowo: ", war)
# odchylenie_stara = odchylenieStandardowe(lista[0])
# print("Odchylenie standardowo nie wektorowo: ", odchylenie_stara)
# odchylenie = odchylenieStandardoweWektorowo(lista[0])
# print("Odchylenie standardowe wektorowo: ", odchylenie)


punkty = []
with open("dane.txt", 'r') as file:
    for line in file:
        tmp = line.split(separator)
        tmp = list(map(lambda i: int(i), tmp))
        punkty.append(tmp)
# print(punkty)


def regresja(lista):
    x2 = [row[0] for row in lista]
    x2 = np.array(x2)
    x1 = np.ones(len(x2), dtype=np.int8)
    x1 = x1.reshape((1, len(x1)))
    x2 = x2.reshape((1, len(x2)))
    x = np.concatenate((x1, x2))
    x = np.transpose(x)
    y = [row[1] for row in punkty]
    inversedX = np.linalg.inv(np.matmul(x.T, x))
    b = np.matmul(inversedX, np.matmul(x.T, y))
    return b


# b = regresja(punkty)
# print(b)


def wektorKolumnowy(lista, i):
    wektor = []
    for j in range(len(lista)):
        wektor += [lista[j][i]]
    return np.array(wektor)


def projekcja(u, v):
    L = np.dot(u, np.transpose(v))
    M = np.dot(u, np.transpose(u))
    projekcja = (L / M) * u
    return projekcja


def dekompozycja(lista):
    listaU = []
    listaE= []
    for i in range(len(lista[0])):
        if i == 0:
            listaU += [wektorKolumnowy(lista, i)]
        else:
            proj = 0
            for y in range(i):
                proj += projekcja(listaU[y], wektorKolumnowy(lista, i))
            listaU += [wektorKolumnowy(lista, i) - proj]
    listaU = np.transpose(listaU)
    for j in range(len(lista[0])):
        norma = math.sqrt(np.dot(wektorKolumnowy(listaU, j), wektorKolumnowy(listaU, j)))
        listaE += [1/norma * wektorKolumnowy(listaU, j)]
    listaE = np.transpose(listaE)
    # R = np.dot(np.transpose(listaE), lista)
    # R = np.matrix.round(R, 3)
    return listaE


niezalezneWektory = np.array([[2, 0],
                              [0, 1],
                              [0, 2]])


# Q, R = dekompozycja(niezalezneWektory)
# R = dekompozycja(Q)
# print("Mcierz Q:\n", Q)
# print("Macierz R:\n", R)


def isuppertriangular(lista):
    flag = 0
    lista = np.matrix.round(lista, 3)
    for i in range(1, len(lista)):
        for j in range(0, i):
            if lista[i][j] != 0:
                flag = 0
            else:
                flag = 1
    if flag == 1:
        return True
    else:
        return False


def Ak(lista):
    Q = dekompozycja(lista)
    lista2 = np.dot(np.transpose(Q), lista)
    lista2 = np.dot(lista2, Q)
    return lista2


def wartosciwlasne(lista):
    lista2 = lista
    while not isuppertriangular(lista2):
        lista2 = Ak(lista2)
    return np.diag(lista2)


a = np.array([[3, 2],
              [4, 1]])


def odejmijodprzekatnej(lista, wartoscwlasna):
    listakopia = copy.deepcopy(lista)
    for i in range(len(lista)):
        for j in range(len(lista[0])):
            if i == j:
                listakopia[i][j] = listakopia[i][j] - wartoscwlasna
    return listakopia


def macierzewlasne(lista, wartosciwlasne):
    wynik = {}
    zera = np.zeros((len(lista), 1))
    j = 0
    for i in wartosciwlasne:
        wynik[j] = odejmijodprzekatnej(lista, i)
        wynik[j] = np.hstack((wynik[j], zera))
        j += 1
    return wynik


def printmacierze(lista):
    for i in lista:
        print(lista[i])
        print("\n")


wartosci_wlasne = wartosciwlasne(a)
wartosci_wlasne = np.matrix.round(wartosci_wlasne, 3)
print("Macierz \n", a)
print("Wartosci własne macierzy\n", wartosci_wlasne)
slownik = macierzewlasne(a, wartosci_wlasne)
print("Macierze po odjeciu wartosci własnych z przekatnej\n")
printmacierze(slownik)


def wektorywlasne(lista):
    for i in range(len(lista)):
        if lista[i][i] == 0.0:
            print('Dzielenie przez 0')
            return 1
        for j in range(len(lista)):
            if i != j:
                ratio = lista[j][i] / lista[i][i]
                for k in range(len(lista) + 1):
                    lista[j][k] = lista[j][k] - ratio * lista[i][k]
    return [lista[i][len(lista)]/lista[i][i] for i in range(len(lista))]

# print("Wektory wlasne macierzy:\n")
# for i in slownik:
#     # print("PRZED\n", slownik[i])
#     slownik[i] = np.delete(slownik[i], len(slownik)-1, 0)
#     # print("PO\n", slownik[i])
#     print(np.round(wektorywlasne(slownik[i])+[-1.], 4)*-1)
