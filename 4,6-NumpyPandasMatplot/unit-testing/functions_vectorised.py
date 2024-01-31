import numpy as np

def rletranslator(x):
    xs=np.array([])
    for i in range(x.shape[1]):
        xs=np.append(xs,np.array(x[0,i]*np.ones(x[1,i])))
    return xs

def sum_non_neg_diag(X: np.ndarray) -> int:

    res=0
    dia=(np.diag(X))
    if np.all(dia<0):
        return -1
    else:
        for i in np.where(dia>0):
            res+=dia[i]
        return sum(res)


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:

    x.sort()
    y.sort()
    return (np.array_equal(x, y))


def max_prod_mod_3(x: np.ndarray) -> int:

    if len(x)>1:
        lstprod= (x*np.concatenate((x[-1:], x[:-1])))[1:]
        mask = lstprod % 3 == 0
        return np.amax(mask*lstprod) if np.any(mask) else -1
    else:
        return -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    shap=image.shape
    res=np.zeros((shap[0],shap[1]))
    for i in range(len(image)):
        res[i] = np.sum(image[i]*weights,axis=1)
    return res


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
        xx=x.T
        yy=y.T
        if np.sum(xx[1],axis=0)==np.sum(yy[1],axis=0):
            xs=np.array([])
            for i in range(xx.shape[1]):
                xs=np.append(xs,np.array(xx[0,i]*np.ones(x.T[1,i])))
            ys=np.array([])
            for i in range(yy.shape[1]):
                ys=np.append(ys,np.array(yy[0,i]*np.ones(y.T[1,i])))
            return int(np.sum(xs*ys))
        else:
            return -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    DotProducts = X.dot(Y.T)
    xnorm = np.array([np.linalg.norm(X, axis=1)])
    ynorm = np.array([np.linalg.norm(Y, axis=1)]).T
    CosineSimilarity = DotProducts / (xnorm*ynorm).T
    for i in np.where(xnorm==0):
            CosineSimilarity[i]=np.ones(CosineSimilarity.shape[1])
    for i in np.where(ynorm==0):
            (CosineSimilarity.T)[i]=np.ones(CosineSimilarity.shape[0])
    return CosineSimilarity
