import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.linalg import lstsq, inv
from scipy.ndimage import convolve1d

def savitzky_golay(x, okno, pol, deriv=0, state="Classic"):
    #получаем коэф, перемножаем окно на коэф - записываем число, смещение
    x = np.array(x).astype(np.float64)
    n = x.size
    if (okno > n):
        raise Exception("Нельзя так! Величина окна больше количества точек.")

    if state not in ("Classic", "Chinese"):
        raise Exception("Неправильно задан параметр state! (Classic, Chinese)")

    if deriv > pol:
        coeffs = np.zeros(okno)
    else:
        m, ost = divmod(okno, 2)
        if ost == 0:
            z = m - 0.5  #она же i в ориг статье, но для нечет
        else:
            z = m

        x1 = np.arange(-z, okno - z, dtype=float)
        x1 = x1[::-1]
        order = np.arange(pol + 1).reshape(-1, 1)
        V = x1 ** order  #значит Вендетта... (Вандермонд) (S(trans) в 2005)

        if state=="Classic":
            y = np.zeros(pol + 1)
            if deriv == 0:
                y[0] = 1
            else:
                y[deriv] = deriv  #тут вообще der!/h^der
            coeffs, _, _, _ = lstsq(V, y)  # остальное это residues, rank, min(M,N)
        else:
            # Коэффициенты по статье 2005
            S = np.dot(V, V.transpose())
            C = np.dot(V.transpose(), inv(S))
            C = C.transpose()
            if deriv == 0:
                coeffs = C[deriv]
            else:
                coeffs = C[deriv]*deriv


    #увеличиваем x на окно слева\справа (классика - не трогаем x)
    #x = np.concatenate((x[m - 1::-1], x, x[-1:-1 - m:-1]))
    # "ручная" конволюция
    #for i in range(m, m + n):
    #    x[i] = np.dot(x[(i - m):(i + m + ost)], coeffs)
    #x = x[m:m + n]
    #return pd.Series(x)
    y = convolve1d(x, coeffs)
    return pd.Series(y)

filename = ''
for file in glob.glob('*'):
    print(f"Файл: {file}. Это нужный файл?")
    s = input("Да = y/д, Нет - любая клавиша... ")
    if s in ('y', 'д'):
        filename = file
        break
if filename == '':  exit(print('Других файлов не найдено!'))

savgol_param = None
while savgol_param is None:
    print('Задайте параметры фильтра Савицкого-Голэя в виде: ширина окна, степень полинома, порядок производной:')
    try:
        savgol_param = list(map(int, input().split(',')))
        if len(savgol_param) != 3 or savgol_param[1] > savgol_param[0]:
            savgol_param = None
            print("Неправильно заданы параметры!")
    except:
        print("Неправильно заданы параметры!")
        savgol_param = None

def median_filter(x, okno):
    """
    Для расчета крайних точек:
    ‘reflect’ (d c b a | a b c d | d c b a)     ‘constant’ (k k k k | a b c d | k k k k)
    ‘nearest’ (a a a a | a b c d | d d d d)
    """
    if okno == 1:
        return x

    x = np.array(x).astype(np.float64)
    n = x.size
    if n < okno or okno < 1:
        raise Exception("Неподходящий размер окна!")

    m, ost = divmod(okno, 2)
    #x = np.concatenate((x[m - 1::-1], x, x[-1:-1 - m:-1])) #reflect
    x = np.concatenate((np.zeros(m, dtype=float), x, np.zeros(m, dtype=float)))

    for i in range(m, m + n):
        x1 = x[(i - m):(i + m + ost)]
        x1 = np.sort(x1)
        if ost == 1:
            x[i] = x1[m]
        else:
            x[i] = (x1[m - 1] + x1[m]) / 2

    x = x[m:m + n]
    return pd.Series(x)

df = pd.read_csv(filename, header=None, sep='\\s+')
df = df.transpose()

median_param = int(input("Задайте окно для коррекции базовой линии: "))
pokaz = None
while pokaz is None:
    print(f"Всего {len(df.columns)-1} образцов. Введите номер или диапазон для отображения в формате 5 или 5,10:")
    try:
        pokaz = list(map(int, input().split(",")))
        print(pokaz)
        if (len(pokaz) not in (1, 2)):
            print("Неправильно задан диапазон!")
            pokaz = None
        elif len(pokaz) == 2:
            if (pokaz[0] > pokaz[1]) or (pokaz[1] not in df.columns):
                print("Неправильно задан диапазон!")
                pokaz = None
    except:
        print("Неправильно задан диапазон!")
        pokaz = None
#добавляем все фильтрованные данные в массив, если нужно - выводим
df_out = pd.DataFrame(df.iloc[:, 0])
df_out_b = pd.DataFrame(df.iloc[:, 0])

fig1 = plt.figure("Обработка фильтром Савицкого-Голэя", constrained_layout=True)
fig2 = plt.figure("Коррекция базовой линии", constrained_layout=True)

cmap = plt.get_cmap('jet')
if len(pokaz) == 2:
    colors = cmap(np.linspace(0, 1.0, pokaz[1] - pokaz[0] + 1))
    k = 0

start_time = time()
for i in df.columns:
    if i == 0: continue
    y1 = savitzky_golay(df.iloc[:, i], *savgol_param, state="Chinese")
    df_out[len(df_out.columns)] = y1
    y_base = median_filter(y1, median_param)
    y_out = y1 - y_base
    df_out_b[len(df_out_b.columns)] = y_out

    if (pokaz[0] == i) or (len(pokaz) == 2 and (i in range(pokaz[0], pokaz[1]+1))):
        if len(pokaz) == 2:
            color = colors[k]
            k += 1
        else:
            color = 'r'
        plt.figure(fig1)
        ax = plt.subplot(211)
        ax.title.set_text("Исходные данные") #plt.sublot.title.set_text
        ax.title.set_fontsize(8)
        plt.plot(df.iloc[:, 0], df.iloc[:, i], color=color, label=str(i))
        plt.legend(ncols=3)
        ax = plt.subplot(212); ax.title.set_text("Фильтр Савицкого-Голэя"); ax.title.set_fontsize(8)
        plt.plot(df.iloc[:, 0], y1, color = color)

        # fig 2 - raw, y_base, y_out
        plt.figure(fig2)
        ax = plt.subplot(311)
        ax.set_title("Исходные данные"); ax.title.set_fontsize(8)
        plt.plot(df.iloc[:, 0], df.iloc[:, i], color=color, label=str(i))
        plt.legend(ncols=3)
        ax = plt.subplot(312)
        ax.set_title("Базовая линия (медиана)"); ax.title.set_fontsize(8)
        plt.plot(df.iloc[:, 0], y_base, color=color)
        ax = plt.subplot(313)
        ax.title.set_text("Вычет базовой"); ax.title.set_fontsize(8)
        plt.plot(df.iloc[:, 0], y_out, color=color)

plt.figure(fig1)
plt.show()

file = input("Готово! Введите имя файла для сохранения или введите н/n: ")
if file not in ("n", "н"):
    df_out = df_out.transpose()
    df_out_b = df_out_b.transpose()
    df_out.to_csv(f"{file}.txt", header=None, index=None, sep=' ')
    df_out_b.to_csv(f"{file}_wbase.txt", header=None, index=None, sep=' ')
