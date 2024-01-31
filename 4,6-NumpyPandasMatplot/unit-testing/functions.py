from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    sum=0
    flag=0
    if all([len(e)==len(X[0]) for e in X]):
        for i in range(len(X[0])):
            if X[i][i] >= 0:
                sum+=X[i][i]
            else:
                flag-=1
        return -1 if sum==0 and flag<0 else sum
    else:
        return -1



def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x.sort()
    y.sort()
    return x==y



def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    maxmb=[]
    for i in range(1,len(x)):
        if x[i-1]*x[i]%3==0:
            maxmb.append(x[i-1]*x[i])
    return max(maxmb) if len(maxmb)>0 else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res=[]
    restemp=[]
    sumchan=0
    for i in range(len(image)):                 # выбор матрицы
        for j in range(len(image[i])):          # выбор строки матрицы
            for k in range(len(image[i][j])):   # работа с строкой
                sumchan+=image[i][j][k]*weights[k]
            restemp.append(sumchan)
            sumchan=0
        res.append(restemp)
        restemp=[]
    return res




def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    xs=""
    ys=""
    res=[]
    for i in range(len(x)):
        xs+=str(x[i][0])*x[i][1]
    for i in range(len(y)):
        ys+=str(y[i][0])*y[i][1]
    if len(xs)==len(ys):
        for i in range(len(xs)):
            res.append(int(xs[i])*int(ys[i]))
        return sum(res)
    else:
        return -1


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res=[]
    resbuf=[]
    for i in range(len(X)):
        dx=sum([l*l for l in X[i]])**0.5
        if dx==0:
            res.append([1.0 for k in range(len(X[0]))])
            continue

        for j in range(len(Y)):
            dy=sum([l*l for l in Y[j]])**0.5
            if dy==0:
                resbuf.append(1)
                break
            else:
                resbuf.append(sum([X[i][k]*Y[j][k]/(dx*dy) for k in range(len(X[0]))]))
        res.append(resbuf)
        resbuf=[]
    return res
