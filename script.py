import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    mu = np.zeros((5, 2))                       # create a matrix to hold mean for 5 different class
    count = [0, 0, 0, 0, 0]                     # count how many examples exist in each class
    for i in range(len(X)):
        if y[i] == 1:
            mu[0, :] = mu[0, :] + X[i, :]       # sum up values for class 1
            count[0] = count[0] + 1
        if y[i] == 2:
            mu[1, :] = mu[1, :] + X[i, :]       # sum up values for class 2
            count[1] = count[1] + 1
        if y[i] == 3:
            mu[2, :] = mu[2, :] + X[i, :]       # sum up values for class 3
            count[2] = count[2] + 1
        if y[i] == 4:
            mu[3, :] = mu[3, :] + X[i, :]       # sum up values for class 4
            count[3] = count[3] + 1
        if y[i] == 5:
            mu[4, :] = mu[4, :] + X[i, :]       # sum up values for class 5
            count[4] = count[4] + 1

    means = np.zeros((2, 5))
    means[:, :] = np.divide(np.transpose(mu), count)    # mean = total value / number of examples in a class
    covmat = np.cov(X.T)                                # calculating covariance for the entire given data X
    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # IMPLEMENT THIS METHOD

    class1 = np.zeros((1, 2))
    class2 = np.zeros((1, 2))
    class3 = np.zeros((1, 2))
    class4 = np.zeros((1, 2))
    class5 = np.zeros((1, 2))
    countt = [0, 0, 0, 0, 0]

    for i in range(len(X)):
        if y[i] == 1:
            class1 = np.append(class1, X[i, :])         # extracting examples belong to class 1
            countt[0] = countt[0] + 1
        if y[i] == 2:
            class2 = np.append(class2, X[i, :])         # extracting examples belong to class 2
            countt[1] = countt[1] + 1
        if y[i] == 3:
            class3 = np.append(class3, X[i, :])         # extracting examples belong to class 3
            countt[2] = countt[2] + 1
        if y[i] == 4:
            class4 = np.append(class4, X[i, :])         # extracting examples belong to class 4
            countt[3] = countt[3] + 1
        if y[i] == 5:
            class5 = np.append(class5, X[i, :])         # extracting examples belong to class 5
            countt[4] = countt[4] + 1

    class1 = np.reshape(class1[2:], (countt[0], 2))
    class2 = np.reshape(class2[2:], (countt[1], 2))
    class3 = np.reshape(class3[2:], (countt[2], 2))
    class4 = np.reshape(class4[2:], (countt[3], 2))
    class5 = np.reshape(class5[2:], (countt[4], 2))

    # calculating covariances for different classes of examples
    covmats = [np.cov(class1.T), np.cov(class2.T), np.cov(class3.T), np.cov(class4.T), np.cov(class5.T)]
    # means stay the same as in ldaLearn
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD

    '''
    # applying posterior density function with gaussian distribution with one cov value regarded to the entire data Xtest
    '''

    # calculating X - mu and its transpose
    means = means.T
    xMinusMu1 = Xtest - means[0, :]
    xMinusMu2 = Xtest - means[1, :]
    xMinusMu3 = Xtest - means[2, :]
    xMinusMu4 = Xtest - means[3, :]
    xMinusMu5 = Xtest - means[4, :]
    xMinusMus = [xMinusMu1, xMinusMu2, xMinusMu3, xMinusMu4, xMinusMu5]               # X - mu
    xMinusMusT = [xMinusMu1.T, xMinusMu2.T, xMinusMu3.T, xMinusMu4.T, xMinusMu5.T]    # Transpose(X - mu)

    # calculating sigma^(-1)
    sigInv = inv(covmat)

    # apply e^(- (Transpose(x - mu) inv(sigma) (x - mu)) / 2)
    # p(y) and denominator are the same for every class
    numerators = np.zeros((len(Xtest), 5))
    for outeri in range(5):
        for inneri in range(len(Xtest)):
            temp = np.dot(xMinusMus[outeri][inneri, :], sigInv)
            numerators[inneri, outeri] = np.dot(temp, xMinusMusT[outeri][:, inneri])
            numerators[inneri, outeri] = np.divide(numerators[inneri, outeri], 2)
            numerators[inneri, outeri] = np.exp(-numerators[inneri, outeri])

    '''
    # predicting examples by the class which has the largest posterior density value
    '''
    ypred = np.argmax(numerators, axis=1)
    ones = np.ones(ypred.shape)
    ypred = np.add(ypred, ones)
    ypred = ypred.reshape(ytest.shape)
    countCorrectPred = 0
    for i in range(len(ypred)):
        if ypred[i, :] == ytest[i, :]:
            countCorrectPred = countCorrectPred + 1
    acc = countCorrectPred / len(ytest)          # acc = number of correct predictions / total number of examples

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD

    '''
    # applying posterior density function with gaussian distribution using different cov values for different classes
    '''

    # computer the fixed denominator values for each class
    # 1 / determinant(sigma)^(1/2)
    # ignoring 1 / (2*pi)^(D/2) since this value is same for every class
    denominators = [np.sqrt(det(covmats[0])), np.sqrt(det(covmats[1])), np.sqrt(det(covmats[2])),
                    np.sqrt(det(covmats[3])), np.sqrt(det(covmats[4]))]

    # calculating X - mu and its transpose
    means = means.T
    xMinusMu1 = Xtest - means[0, :]
    xMinusMu2 = Xtest - means[1, :]
    xMinusMu3 = Xtest - means[2, :]
    xMinusMu4 = Xtest - means[3, :]
    xMinusMu5 = Xtest - means[4, :]
    xMinusMus = [xMinusMu1, xMinusMu2, xMinusMu3, xMinusMu4, xMinusMu5]               # X - mu
    xMinusMusT = [xMinusMu1.T, xMinusMu2.T, xMinusMu3.T, xMinusMu4.T, xMinusMu5.T]    # Transpose(X - mu)

    # calculating sigma^(-1)
    sigInvs = [inv(covmats[0]), inv(covmats[1]), inv(covmats[2]), inv(covmats[3]), inv(covmats[4])]

    # apply denominator * e^(- (Transpose(x - mu) inv(sigma) (x - mu)) / 2)
    # p(y) is the same for every class
    numerators = np.zeros((len(Xtest), 5))
    for outeri in range(5):
        for inneri in range(len(Xtest)):
            temp = np.dot(xMinusMus[outeri][inneri, :], sigInvs[outeri])
            numerators[inneri, outeri] = np.dot(temp, xMinusMusT[outeri][:, inneri])
            numerators[inneri, outeri] = np.divide(numerators[inneri, outeri], 2)
            numerators[inneri, outeri] = np.exp(-numerators[inneri, outeri])
            numerators[inneri, outeri] = np.divide(numerators[inneri, outeri], denominators[outeri])

    '''
    predicting examples by the class which has the largest posterior density value
    '''
    ypredd = np.argmax(numerators, axis=1)
    ones = np.ones(ypredd.shape)
    ypredd = np.add(ypredd, ones)
    ypredd = ypredd.reshape(ytest.shape)
    countCorrectPredd = 0
    for i in range(len(ypredd)):
        if ypredd[i, :] == ytest[i, :]:
            countCorrectPredd = countCorrectPredd + 1
    accc = countCorrectPredd / len(ytest)          # acc = number of correct predictions / total number of examples

    return accc, ypredd


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD

    '''
    The weights are learned using equation w = (transpose(X).X)^(-1).transpose(X).y. This equation was obtained from
    Part C, Handout 1. It is equation used for learning parameters for linear regression.
    '''
    xty = np.dot(X.T, y)                            #Transpose(X).X
    xtx = np.dot(X.T, X)                            #Transpose(X).y
    w = np.dot(np.linalg.inv(xtx), xty)             #w = (transpose(X).X)^(-1).transpose(X).y

    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD

    '''
    The weights are learned using equation w = (lambd*I+transpose(X).X)^(-1).transpose(X).y. This equation was obtained from
    Part C, Handout 1. It is equation used for learning parameters for ridge regression.
    '''
    temp_one = (X.T).dot(X)                                 #transpose(X).X
    arr = np.zeros((len(X[0, :]), len(X[0, :])), int)       #make zero array of size similar to X
    np.fill_diagonal(arr, 1)                                #make diagonal to 1 ->I
    temp = lambd * arr                                      #lambd*I
    temp_four = temp + temp_one                             #lambd*I + transpose(X).X
    temp_two = np.linalg.inv(temp_four)                     #inverse of (lambd*I + transpose(X).X)
    temp_three = (X.T).dot(y)                               #transpose(X).y
    w = temp_two.dot(temp_three)                            #w = (lambd*I+transpose(X).X)^(-1).transpose(X).y

    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    # IMPLEMENT THIS METHOD
    '''
    This function is used to apply learnt weights from learnOLERegression function on training and test data and
    calculate the Mean Squared Error.
    The Equation used to implement MSE is MSE = (1/N).summation(y-transpose(w).x)^2
    '''

    result = 0
    for i in range(len(Xtest)):
        temp = ytest[i] - np.dot(Xtest[i, :], w)        #y-x.w = y - (w.T).x
        result = result + temp * temp                   #summation of squared quantities
    mse = result / len(Xtest)                           #MSE = (1/N).summation(y-transpose(w).x)^2
    return mse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    # IMPLEMENT THIS METHOD
    '''
    This function is used to apply learnt weights from learnRidgeRegression function on training and test data and
    calculate the error and gradiant of error.
    The Equation used to implement error is error = (1/2)summation(y-transpose(w).x)^2 + (1/2)lambd.transpose(w).w
    obtained from project description (4)
    The Equation used to implement error gradient is error_grad = d(J(w))/dw + lambd.w
    '''
    w = np.reshape(w, (1, len(X.T)))
    y = np.reshape(y, (len(X), 1))
    lambdWTW = 0.5 * lambd * np.dot(w, w.T).flatten()[0]            #(1/2)lambd.transpose(w).w
    wtx = np.dot(X, w.T)                                            #transpose(w).x
    InOutDiff = np.subtract(y, wtx)                                 #y-transpose(w).x

    InOutDiff2 = np.multiply(InOutDiff, InOutDiff)                  #squaring the y-transpose(w).x
    sumIOD2 = np.sum(InOutDiff2, axis=0)                            #summation of squared term
    error = 0.5 * sumIOD2.flatten()[0] + lambdWTW                   #error = (1/2)summation(y-transpose(w).x)^2 + (1/2)lambd.transpose(w).w
    #print("error", error)

    grad = np.dot(InOutDiff.T, np.subtract(0, X))                   #transpose(y-transpose(w).x).-X
    lambdW = np.multiply(w, lambd)                                  #lambd*w
    error_grad = grad.flatten() + lambdW.flatten()                  #error_grad = d(J(w))/dw + lambd.w
    '''
    #Another form of implementation which slightly run for longer time but gives same graph. (Not currently used)
    lh = (w.T).dot(w)
    lhh = 0.5 * lambd * lh
    summ = 0

    for i in range(len(X)):
        temp = y[i] - ((w.T).dot(X[i]))  # changing dot to * changes shape from (65,) to (65,65)
        summ = summ + (temp ** 2)
    summ = summ / 2
    jw = summ + lhh
    error = jw
    print(error)

    temp_two = (X.T).dot(X)
    temp_three = temp_two.dot(w)
    temp_three = temp_three.reshape(65, 1)
    temp_four = (X.T).dot(y)
    temp_five = lambd * w
    temp_five = temp_five.reshape(65, 1)
    error_grad = temp_three.flatten() - temp_four.flatten() + temp_five.flatten()
    '''
    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    '''
    This function is used to implement non-linear regression part to convert single attribute x to p attributes
    Xd = [1,x,x^2,..,x^p]
    '''
    x = x.reshape(x.size, 1)
    Xd = np.ones((x.size, 1))                       #Xd = [1]
    for i in range(1, p + 1):                       #loop until value of p+1 to incorporate p
        powx = np.power(x, i)                       #x^i where i goes from 0 to p
        Xd = np.concatenate((Xd, powx), axis=1)     #Continously concatenate in Xd the x^i
    return Xd


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))

# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, X, y)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, X_i, y)

print('MSE without intercept for training data:' + str(mle))
print('MSE with intercept for training data:' + str(mle_i))

mle = testOLERegression(w, Xtest, ytest)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('MSE without intercept for test data' + str(mle))
print('MSE with intercept for test data:' + str(mle_i))
print('Relative Weight for OLE' + str(np.sqrt(sum(w_i**2))))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    if(lambd==0):
        print('MSE train for lambd = 0:',mses3_train[i])
        print('MSE test for lambd = 0:',mses3[i])
    if(lambd==0.06):
        print('MSE train for lambd = 0.06:', mses3_train[i])
        print('MSE test for lambd = 0.06:', mses3[i])
    if (lambd == 1):
        print('MSE train for lambd = 1:', mses3_train[i])
        print('MSE test lambd = 1:', mses3[i])
    i = i + 1
print('Relative Weight for Ridge Regression:' + str(np.sqrt(np.sum(w_l**2))))
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 100}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    if (lambd == 0):
        print('MSE train for lambd = 0 using gradient descent:', mses4_train[i])
        print('MSE test lambd = 0 using gradient descent:', mses4[i])
    if (lambd == 0.06):
        print('MSE train for lambd = 0.06 using gradient descent:', mses4_train[i])
        print('MSE test for lambd = 0.06 using gradient descent:', mses4[i])
    if (lambd == 1):
        print('MSE train lambd = 1 using gradient descent:', mses4_train[i])
        print('MSE test for lambd = 1 using gradient descent:', mses4[i])
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    print('p:',p,' error for lambd = 0:',mses5[p,0])
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    print('p:', p, ' error for lambd = 0.06:', mses5[p,1])
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))

plt.show()