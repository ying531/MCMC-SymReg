import pandas as pd
import numpy as np

def f1(n_train=100,n_test=30):
    """
    生成模拟数据
    y = x0*x1 + sin((x0-1)*(x1+1))
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    x1 = np.random.uniform(0.1, 5.9, n_train)
    x2 = np.random.uniform(0.1, 5.9, n_train)
    x1 = pd.DataFrame(x1)
    x2 = pd.DataFrame(x2)
    Xtrain = pd.concat([x1, x2], axis=1)
    ytrain = Xtrain.iloc[:,0]*Xtrain.iloc[:,1]+np.sin((Xtrain.iloc[:,0]-1)*(Xtrain.iloc[:,1]+1))

    xx1 = [0]*30
    xx2 = [0]*30
    for i in np.arange(30):
        for j in np.arange(30):
            xx1.append(-0.15 + 0.2 * i)
            xx2.append(-0.15 + 0.2 * j)
    xx1 = pd.DataFrame(xx1)
    xx2 = pd.DataFrame(xx2)
    Xtest = pd.concat([xx1, xx2], axis=1)
    ytest = Xtest.iloc[:, 0] * Xtest.iloc[:, 1] + np.sin((Xtest.iloc[:, 0] - 1) * (Xtest.iloc[:, 1] + 1))
    return Xtrain, ytrain, Xtest, ytest

def f2(n_train=100,n_test=30):
    """
    生成模拟数据
    y = x0^4 - x0^3 + x1^2 -x1
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0, 6, (n_train, 2))
    ytrain = Xtrain[:, 0]**4 - Xtrain[:, 0]**3 + Xtrain[:, 1]**2 - Xtrain[:, 1]

    Xtest = np.random.uniform(0, 6, (n_test,2))
    ytest = Xtest[:, 0] ** 4 - Xtest[:, 0] ** 3 + Xtest[:, 1] ** 2 - Xtest[:, 1]

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)

    return Xtrain, ytrain, Xtest, ytest


def f3(n_train=100,n_test=30):
    """
    生成模拟数据
    y = 6 * sin(x0) * cos(x1)
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0,6,(n_train,2))
    ytrain = 6*np.sin(Xtrain[:,0])*np.cos(Xtrain[:,1])

    Xtest = np.random.uniform(0, 6, (n_test, 2))
    ytest = 6 * np.sin(Xtest[:, 0]) * np.cos(Xtest[:, 1])

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest
def f4(n_train=100,n_test=30):
    """
    生成模拟数据
    y = 15 + 2 * x0^2 + 2* x1^3
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0, 6, (n_train, 2))
    ytrain = 15 + 2 * Xtrain[:,0]**2 + 2* Xtrain[:,1]**3

    Xtest = np.random.uniform(0, 6, (n_test, 2))
    ytest = 15 + 2 * Xtest[:, 0] ** 2 + 2 * Xtest[:, 1] ** 3
    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest

# def f5(n_train=100,n_test=30):
#     """
#     生成模拟数据
#     y = x0^3/5 + x1^3/2 - x1 - x0
#     :param n_train:train set大小
#     :param n_test: test set大小
#     :return: Xtrain, ytrain, Xtest, ytest
#     """
#     Xtrain = np.random.uniform(0, 6, (n_train, 2))
#
#
#     Xtrain = pd.DataFrame(Xtrain)
#     ytrain = pd.Series(ytrain)
#     Xtest = pd.DataFrame(Xtest)
#     ytest = pd.Series(ytest)
#     return Xtrain, ytrain, Xtest, ytest

def f6(n_train=100,n_test=30):
    """
    生成模拟数据
    y = 0.6 * x0 - 0.85
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0,6,(n_train,1))
    ytrain = 0.6*Xtrain[:,0] - 0.85

    Xtest = np.random.uniform(0,6,(n_test,1))
    ytest = 0.6 * Xtest[:, 0] - 0.85

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest

# def f7(n_train=100,n_test=30):
#     """
#     生成模拟数据
#     y = 0.22 * x0^2 + 0.05
#     :param n_train:train set大小
#     :param n_test: test set大小
#     :return: Xtrain, ytrain, Xtest, ytest
#     """
#
#     Xtrain = pd.DataFrame(Xtrain)
#     ytrain = pd.Series(ytrain)
#     Xtest = pd.DataFrame(Xtest)
#     ytest = pd.Series(ytest)
#     return Xtrain, ytrain, Xtest, ytest
# def f8(n_train=100,n_test=30):
#     """
#     生成模拟数据
#     y = 0.17*x0*x1 + 0.369 * x1^2 - 0.3
#     :param n_train:train set大小
#     :param n_test: test set大小
#     :return: Xtrain, ytrain, Xtest, ytest
#     """
#     Xtrain = pd.DataFrame(Xtrain)
#     ytrain = pd.Series(ytrain)
#     Xtest = pd.DataFrame(Xtest)
#     ytest = pd.Series(ytest)
#     return Xtrain, ytrain, Xtest, ytest
def f9(n_train=100,n_test=30):
    """
    生成模拟数据
    y = 0.49 * x0 * x1^3 - 0.08 * x0^4 +0.36
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0,6,(n_train,2))
    ytrain = 0.49*Xtrain[:,0]*Xtrain[:,1]**3 - 0.08*Xtrain[:,0]**4 + 0.36

    Xtest = np.random.uniform(0, 6, (n_test, 2))
    ytest = 0.49 * Xtest[:, 0] * Xtest[:, 1] ** 3 - 0.08 * Xtest[:, 0] ** 4 + 0.36

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest
def f10(n_train=100,n_test=30):
    """
    生成模拟数据
    y = 0.35*x2 - 0.32*x1*x2 - 0.35 *x0*x1^2 - 0.39*x2^4 + 0.24
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0, 6, (n_train, 3))
    ytrain = 0.35*Xtrain[:,2] - 0.32*Xtrain[:,1]*Xtrain[:,2] - \
             0.35*Xtrain[:,0]*Xtrain[:,1]**2 - 0.39*Xtrain[:,2]**4 + 0.24

    Xtest = np.random.uniform(0, 6, (n_test, 3))
    ytest = 0.35 * Xtest[:, 2] - 0.32 * Xtest[:, 1] * Xtest[:,2] - \
             0.35 * Xtest[:, 0] * Xtest[:, 1] ** 2 - 0.39 * Xtest[:, 2] ** 4 + 0.24

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest

# 下面是一些非additive形式的函数形式
def f11(n_train=100,n_test=30):
    """
    生成模拟数据
    y = exp(-(x0-1)^2)/(1.2+(x1-2.5)^2)
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0,6,(n_train, 2))
    ytrain = np.exp(-(Xtrain[:,0]-1)**2)/(1.2+(Xtrain[:,1]-2.5)**2)

    Xtest = np.random.uniform(0, 6, (n_test, 2))
    ytest = np.exp(-(Xtest[:, 0] - 1) ** 2) / (1.2 + (Xtest[:, 1] - 2.5) ** 2)

    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest

def f12(n_train=100,n_test=30):
    """
    生成模拟数据
    y = exp(-x)*x^3*cos(x)*sin(x)(cos(x)sin^2(x)-1)
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0,6,(n_train,1))
    ytrain = np.exp(-Xtrain[:,0])*Xtrain[:,0]**3*np.cos(Xtrain[:,0])*\
             np.sin(Xtrain[:,0])*(np.cos(Xtrain[:,0]*np.sin(Xtrain[:,0])**2)-1)

    Xtest = np.random.uniform(0, 6, (n_test, 1))
    ytest = np.exp(-Xtest[:, 0]) * Xtest[:, 0] ** 3 * np.cos(Xtest[:, 0]) * \
             np.sin(Xtest[:, 0]) * (np.cos(Xtest[:, 0] * np.sin(Xtest[:, 0]) ** 2) - 1)


    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest
def f13(n_train=100,n_test=30):
    """
    生成模拟数据
    y = f12*(x2-5)
    :param n_train:train set大小
    :param n_test: test set大小
    :return: Xtrain, ytrain, Xtest, ytest
    """
    Xtrain = np.random.uniform(0, 6, (n_train, 2))
    ytrain = np.exp(-Xtrain[:, 0]) * Xtrain[:, 0] ** 3 * np.cos(Xtrain[:, 0]) * \
             np.sin(Xtrain[:, 0]) * (np.cos(Xtrain[:, 0] * np.sin(Xtrain[:, 0]) ** 2) - 1)*(Xtrain[:,1]-5)

    Xtest = np.random.uniform(0, 6, (n_test, 2))
    ytest = np.exp(-Xtest[:, 0]) * Xtest[:, 0] ** 3 * np.cos(Xtest[:, 0]) * \
            np.sin(Xtest[:, 0]) * (np.cos(Xtest[:, 0] * np.sin(Xtest[:, 0]) ** 2) - 1)*(Xtest[:,1]-5)


    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.Series(ytrain)
    Xtest = pd.DataFrame(Xtest)
    ytest = pd.Series(ytest)
    return Xtrain, ytrain, Xtest, ytest

'''
plt.figure()
plt.plot(errList)
plt.title(consle_label)
plt.show()

print("==========MCMC method========")
print("1st:")
print(Express(RootLists[0][-1]))
print("2nd:")
print(Express(RootLists[1][-1]))
print(Beta[0],Beta[1])
#print(Beta[0])
print("==========GP method========")
print(est_gp._program)

'''