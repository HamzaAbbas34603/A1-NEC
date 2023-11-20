from ctypes.wintypes import FLOAT
import numpy as np
import sys
import math
import random
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import timeit

class Dataset:
    nf: int
    no: int
    ns: int
    xtable: list = [] # array of arrays
    ytable: list = [] # array of arrays
    xmin: list
    ymin: list
    xmax: list
    ymax: list

class NN:
    L: int # number of layers
    n: list # array with the number of units in each layer (including the input and output layers)
    h: list # array of arrays for the fields (h)
    xi: list # array of arrays for the activations (ξ)
    w: list # n array of matrices for the weights (w)
    theta: list # array of arrays for the thresholds (θ)
    delta: list # array of arrays for the propagation of errors (Δ)
    d_w: list # array of matrices for the changes of the weights (δw)
    d_theta: list # array of arrays for the changes of the weights (δθ)
    d_w_prev: list # array of matrices for the previous changes of the weights, used for the momentum term (δw (prev))
    d_theta_prev: list # array of arrays for the previous changes of the thresholds, used for the momentum term (δθ(prev))
    fact: str # the name of the activation function that it will be used. It can be one of these four: sigmoid, relu, linear, tanh.

def read_dataset(file_name):
    dataset = Dataset()
    with open(file_name) as f:
        dataset.nf,dataset.no = map(int,f.readline().split(' '))
        f.readline()
        data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
        dataset.ns = len(data)
        for row in data:
            dataset.xtable.append(row[:dataset.nf])
            dataset.ytable.append(row[dataset.nf:dataset.nf+dataset.no])
    return dataset

def print_dataset():
    for m in range(0,dataset.ns):
        for n in range(0,dataset.nf):
            print("{}".format(dataset.xtable[m][n]),end='\t')
        for n in range(0,dataset.no):
            print("{}".format(dataset.ytable[m][n]),end='\t')
        print()

def scale_dataset(s_min,s_max):
    dataset.xmin = np.zeros(dataset.nf)
    dataset.xmax = np.zeros(dataset.nf)
    dataset.ymin = np.zeros(dataset.no)
    dataset.ymax = np.zeros(dataset.no)
    for n in range(0,dataset.nf):
        max = float('-inf')
        min = float('inf')
        for m in range(0,dataset.ns):
            if dataset.xtable[m][n] > max:
                max = dataset.xtable[m][n]
            if dataset.xtable[m][n] < min:
                min = dataset.xtable[m][n]
        dataset.xmin[n] = min
        dataset.xmax[n] = max
        for m in range(0, dataset.ns):
            dataset.xtable[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.xtable[m][n] - min)

    for n in range(0,dataset.no):
        max = float('-inf')
        min = float('inf')
        for m in range(0,dataset.ns):
            if dataset.ytable[m][n] > max:
                max = dataset.ytable[m][n]
            if dataset.ytable[m][n] < min:
                min = dataset.ytable[m][n]
        dataset.ymin[n] = min
        dataset.ymax[n] = max
        for m in range(0,dataset.ns):
            dataset.ytable[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.ytable[m][n] - min)
    return dataset

def descale_dataset(s_min,s_max):
    for n in range(0,dataset.nf):
        min = dataset.xmin[n]
        max = dataset.xmax[n]
        for m in range(0,dataset.ns):
            dataset.xtable[m][n] = min + (max - min)/(s_max - s_min)*(dataset.xtable[m][n] - s_min)
    for n in range(0,dataset.no):
        min = dataset.ymin[n]
        max = dataset.ymax[n]
        for m in range(0,dataset.ns):
            dataset.ytable[m][n] = min + (max - min)/(s_max - s_min)*(dataset.ytable[m][n] - s_min)
    return dataset

def descale_y_value(value,feature,s_min,s_max):
    y_min = dataset.ymin[feature]
    y_max = dataset.ymax[feature]
    return ( y_min + (y_max - y_min)/(s_max - s_min)*(value - s_min) )


def init_nn():
    l = 0
    nn.w = []
    nn.w.append([])
    for i in range(0,nn.n[l]):
        nn.w[l].append([])
        for _ in range(0,dataset.nf):
            nn.w[l][i].append(random.randint(-50,50)/100)
    for l in range(1,nn.L):
        nn.w.append([])
        for i in range(0,nn.n[l]):
            nn.w[l].append([])
            for _ in range(0,nn.n[l-1]):
                nn.w[l][i].append(random.randint(-50,50)/100)

    nn.d_w = []
    l = 0
    nn.d_w.append([])
    for i in range(0,nn.n[l]):
        nn.d_w[l].append([])
        for _ in range(0,dataset.nf):
            nn.d_w[l][i].append(0)

    for l in range(1,nn.L):
        nn.d_w.append([])
        for i in range(0,nn.n[l]):
            nn.d_w[l].append([])
            for _ in range(0,nn.n[l-1]):
                nn.d_w[l][i].append(0)
    
    nn.d_w_prev = []
    l = 0
    nn.d_w_prev.append([])
    for i in range(0,nn.n[l]):
        nn.d_w_prev[l].append([])
        for _ in range(0,dataset.nf):
            nn.d_w_prev[l][i].append(0)
    for l in range(1,nn.L):
        nn.d_w_prev.append([])
        for i in range(0,nn.n[l]):
            nn.d_w_prev[l].append([])
            for _ in range(0,nn.n[l-1]):
                nn.d_w_prev[l][i].append(0)

    nn.theta = []
    for l in range(0,nn.L):
        nn.theta.append([])
        for _ in range(0,nn.n[l]):
            nn.theta[l].append(random.random())
    
    nn.d_theta = []
    for l in range(0,nn.L):
        nn.d_theta.append([])
        for _ in range(0,nn.n[l]):
            nn.d_theta[l].append(0)

    nn.d_theta_prev = []
    for l in range(0,nn.L):
        nn.d_theta_prev.append([])
        for _ in range(0,nn.n[l]):
            nn.d_theta_prev[l].append(0)

    l=0
    nn.h = []
    nn.h.append([])
    for _ in range(0,dataset.nf):
        nn.h[l].append(0)
    for l in range(1,nn.L):
        nn.h.append([])
        for _ in range(0,nn.n[l]):
            nn.h[l].append(0)
    l = 0
    nn.xi = []
    nn.xi.append([])
    for _ in range(0,dataset.nf):
        nn.xi[l].append(0)
    for l in range(1,nn.L):
        nn.xi.append([])
        for _ in range(0,nn.n[l]):
            nn.xi[l].append(0)

    nn.delta = []
    for l in range(0,nn.L):
        nn.delta.append([])
        for _ in range(0,nn.n[l]):
            nn.delta[l].append(0)

def reset_nn():
    l = 0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.w[l][i][j] = random.randint(-50,50)/100

    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.w[l][i][j] = random.randint(-50,50)/100

    l = 0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.d_w[l][i][j] = 0

    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w[l][i][j] = 0

    l = 0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.d_w_prev[l][i][j] = 0

    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w_prev[l][i][j] = 0
    
    for l in range(0,nn.L):
        for i in range(0,nn.n[l]):
            nn.theta[l][i] = random.random()

    for l in range(0,nn.L):
        for i in range(0,nn.n[l]):
            nn.d_theta[l][i] = 0

    for l in range(0,nn.L):
        for i in range(0,nn.n[l]):
            nn.d_theta_prev[l][i] = 0

def feed_forward_propagation():
    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            aux = 0
            for j in range(0,nn.n[l-1]):
                aux += nn.w[l][i][j] * nn.xi[l-1][j]
            nn.h[l][i] = aux - nn.theta[l][i]
            nn.xi[l][i] = a(nn.h[l][i])
    return nn.xi[nn.L-1]

def a(h):
    if activation == 0:
        return 1 / (1 + math.exp(-h))
    if activation == 1:
        return max(h,0)
    if activation == 2:
        # return (2/(1 + math.exp(-2*h))) -1
        return math.tanh(h)

def derivate_a(h):
    if activation == 0:
        g = 1 / (1 + math.exp(-h))
        return (g*(1-g))
    if activation == 1:
        return (0 if h < 0 else 1)
    if activation == 2:
        # return 1 - pow((2/(1 + math.exp(-2*h))) -1,2)
        return 1 - pow(math.tanh(h),2)

def back_propagation(y):
    l = nn.L-1
    for i in range(0,nn.n[l]):
        nn.delta[l][i] = derivate_a(nn.h[l][i]) * (nn.xi[l][i] - y[i])
    for l in range(nn.L-2,0,-1):
        for i in range(0,nn.n[l-1]):
            aux = 0
            for j in range(0,nn.n[l]):
                aux += nn.delta[l][j] * nn.w[l][j][i]
                nn.delta[l-1][i] = derivate_a(nn.h[l-1][i]) * aux

def batched_update_nn(n,alpha):
    for l in range(nn.L-1,0,-1):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w[l][i][j] = -n*nn.delta[l][i]*nn.xi[l-1][j] + alpha * nn.d_w_prev[l][i][j]
                nn.d_w_prev[l][i][j] = nn.d_w[l][i][j]
                nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
            nn.d_theta[l][i] = n*nn.delta[l][i] + alpha * nn.d_theta_prev[l][i]
            nn.d_theta_prev[l][i] = nn.d_theta[l][i]
            nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]

def online_update_nn(n,alpha):
    for l in range(nn.L-1,0,-1):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w[l][i][j] = -n*nn.delta[l][i]*nn.xi[l-1][j] + alpha * nn.d_w_prev[l][i][j]
                nn.d_w_prev[l][i][j] = nn.d_w[l][i][j]
                nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
            nn.d_theta[l][i] = n*nn.delta[l][i] + alpha * nn.d_theta_prev[l][i]
            nn.d_theta_prev[l][i] = nn.d_theta[l][i]
            nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]
    
def partial_update_nn(n,alpha):
    for l in range(nn.L-1,0,-1):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w[l][i][j] = -n*nn.delta[l][i]*nn.xi[l-1][j] + alpha * nn.d_w_prev[l][i][j]
                nn.d_w_prev[l][i][j] = nn.d_w[l][i][j]
                nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
            nn.d_theta[l][i] = n*nn.delta[l][i] + alpha * nn.d_theta_prev[l][i]
            nn.d_theta_prev[l][i] = nn.d_theta[l][i]
            nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]

def main_batched_BP():
    validation_size = trainset_size // 4
    errors = []
    errors1 = []
    for epoch in range(0,epochs):
        error = error1 = 0
        treated = []
        for pat in range(0,trainset_size-validation_size):
            r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            while r in treated:
                r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            nn.xi[0] = dataset.xtable[r]
            treated.append(r)
            feed_forward_propagation()
            back_propagation(dataset.ytable[r])
        batched_update_nn(n,alpha)
        for pat in range(0,trainset_size-validation_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error = error/2
        errors.append(error)
        for pat in range(trainset_size-validation_size,trainset_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error1 += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error1 = error1/2
        errors1.append(error1)
        # print("Epoch: {} \tQuadratic error train set: {}\tQuadratic error validation set: {}".format(epoch, error, error1))
    # plt.plot([i for i in range(0,epochs)], errors, color="blue", linewidth=3)
    # plt.plot([i for i in range(0,epochs)], errors1, color="orange", linewidth=3)
    # plt.show()

def main_online_BP(params):
    validation_size = trainset_size // 4
    errors = []
    errors1 = []
    outFile = open('outputs/errors_{}.csv'.format(params),'w',encoding='utf-8')
    for epoch in range(0,epochs):
        error = error1 = 0
        treated = []
        for pat in range(0,trainset_size-validation_size):
            r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            while r in treated:
                r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            nn.xi[0] = dataset.xtable[r]
            treated.append(r)
            feed_forward_propagation()
            back_propagation(dataset.ytable[r])
            online_update_nn(n,alpha)
        for pat in range(0,trainset_size-validation_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error = error/2
        errors.append(error)
        for pat in range(trainset_size-validation_size,trainset_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error1 += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error1 = error1/2
        errors1.append(error1)
        outFile.write("{}, {}, {}\n".format(epoch, error, error1 ))
    outFile.close()
        # print("Epoch: {} \tQuadratic error train set: {}\tQuadratic error validation set: {}".format(epoch, error, error1))
    # plt.plot([i for i in range(0,epochs)], errors, color="blue", linewidth=3)
    # plt.plot([i for i in range(0,epochs)], errors1, color="orange", linewidth=3)
    # plt.show()

def main_partial_BP():
    validation_size = trainset_size // 4
    errors = []
    errors1 = []
    for epoch in range(0,epochs):
        error = error1 = 0
        treated = []
        cpt = 0
        for pat in range(0,trainset_size-validation_size):
            cpt += 1
            r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            while r in treated:
                r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            nn.xi[0] = dataset.xtable[r]
            treated.append(r)
            feed_forward_propagation()
            back_propagation(dataset.ytable[r])
            if cpt > partial:
                partial_update_nn(n,alpha)
                cpt = 0
        for pat in range(0,trainset_size-validation_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error = error/2
        errors.append(error)
        for pat in range(trainset_size-validation_size,trainset_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error1 += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error1 = error1/2
        errors1.append(error1)
        # print("Epoch: {} \tQuadratic error train set: {}\tQuadratic error validation set: {}".format(epoch, error, error1))
    # plt.plot([i for i in range(0,epochs)], errors, color="blue", linewidth=3)
    # plt.plot([i for i in range(0,epochs)], errors1, color="orange", linewidth=3)
    # plt.show()

def read_epochs():
    with open('epochs.txt') as f:
        ep = []
        te = []
        ve = []
        for i in f.readlines():
            i = i.replace('\n','')
            ep.append(int(i.split('\t')[0]))
            te.append(float(i.split('\t')[1]))
            ve.append(float(i.split('\t')[2]))
    return ep,te,ve

def test_BP_algorithm(trainset_size,s_min,s_max, output_file):
    outFile = open( output_file, "w")
    aux1 = aux2 = 0
    output = []
    for i in range(trainset_size, dataset.ns):
        nn.xi[0] = dataset.xtable[i]
        y_pred = feed_forward_propagation()
        y=z=0
        for j in range(0,dataset.no):
            y += descale_y_value( y_pred[j], j, s_min, s_max )
            z += descale_y_value( dataset.ytable[i][j], j, s_min, s_max )
        output.append(y)
        # print("{}, {}, {}".format(z, y, abs(z - y)))
        outFile.write("{}, {}, {}\n".format(z, y, abs(z - y)))
        aux1 += abs(z - y)
        aux2 += z
    print("\nPercentage of error over the TestSet: {}".format(aux1/aux2*100))
    outFile.close()
    return output,aux1/aux2*100

def run_all_params():
    global activation, epochs, alpha, n, nn
    f = open('params.csv',encoding='utf-8')
    for item in f.readlines()[323:]:
        item = item.replace('\n','').split(',')
        activation = activations.index(item[1])
        epochs = int(item[2])
        n = float(item[3])
        alpha = float(item[4])
        nn.L = int(item[5])
        nn.n = layers[int(item[5])-3]
        init_nn()
        for i in range(0,1):
            print('Running Params ID {} with Test ID {}'.format(item[0],i))
            main_online_BP('{}_{}'.format(item[0],i))
            reset_nn()

def find_best_params(train=False):
    f = open('params.csv',encoding='utf-8')
    errors = []
    for item in f.readlines()[1:]:
        try:
            item = item.replace('\n','').split(',')
            h = open('outputs/errors_{}_0.csv'.format(item[0]))
            last = h.readlines()[-1]
            qt = float(last.replace('\n','').split(',')[1])
            qv = float(last.replace('\n','').split(',')[2])
            errors.append({'qt':qt,'qv':qv,'id':item[0],'params':item[1:]})
            h.close()
        except FileNotFoundError:
            pass
    errors = sorted(errors,key=lambda x:x['qv'])
    for err in errors[:10]:
        print("Training ID: {} \tQuadratic error train set: {}\tQuadratic error validation set: {}\t Params: {}".format(err.get('id'), err.get('qt'), err.get('qv'),err.get('params')))
    
    if train:
        global activation, epochs, alpha, n, nn
        for err in errors[:10]:
            activation = activations.index(err.get('params')[0])
            epochs = int(err.get('params')[1])
            n = float(err.get('params')[2])
            alpha = float(err.get('params')[3])
            nn.L = int(err.get('params')[4])
            nn.n = layers[int(err.get('params')[4])-3]
            init_nn()
            for i in range(0,10):
                print('Running Params ID {} with Test ID {}'.format(err.get('id'),i))
                main_online_BP('_train_{}_{}'.format(err.get('id'),i))
                reset_nn()

def plot_error(id):
    h = open('outputs/errors_{}.csv'.format(id))
    errors = []
    errors1 = []
    epochs = []
    for item in h.readlines():
        item = item.replace('\n','').split(',')
        errors.append(float(item[1]))
        errors1.append(float(item[2]))
        epochs.append(int(item[0]))
    plt.plot(epochs, errors, color="blue", linewidth=3)
    plt.plot(epochs, errors1, color="orange", linewidth=3)
    plt.show()

def order_best_params():
    f = open('params.csv',encoding='utf-8')
    errors = []
    for item in f.readlines()[1:]:
        try:
            item = item.replace('\n','').split(',')
            h = open('outputs/errors_{}_0.csv'.format(item[0]))
            last = h.readlines()[-1]
            qt = float(last.replace('\n','').split(',')[1])
            qv = float(last.replace('\n','').split(',')[2])
            errors.append({'qt':qt,'qv':qv,'id':item[0],'params':item[1:]})
            h.close()
        except FileNotFoundError:
            pass
    errors = sorted(errors,key=lambda x:x['qv'])
    for err in errors[:10]:
        qv = 0
        qt = 0
        for i in range(0,10):
            h = open('outputs/errors__train_{}_{}.csv'.format(err.get('id'),i))
            last = h.readlines()[-1]
            qt += float(last.replace('\n','').split(',')[1])
            qv += float(last.replace('\n','').split(',')[2])
            h.close()
        print("Training ID: {}\tAttempt {}\tQuadratic error train set: {}\tQuadratic error validation set: {}\t Params: {}".format(err.get('id'),i, qt/10, qv/10,err.get('params')))


def main():
    init_nn()

    # start = timeit.default_timer()
    # main_batched_BP()
    # output,result_online = test_BP_algorithm( trainset_size, s_min, s_max, on_test)
    # stop = timeit.default_timer()
    # reset_nn()
    # print('#####  BATCHED BP  #####\tTime (s): ', round(stop - start,2))

    start = timeit.default_timer()
    main_online_BP('orig_0')
    output,result_online = test_BP_algorithm( trainset_size, s_min, s_max, on_test)
    stop = timeit.default_timer()
    reset_nn()
    print('#####  ONLINE BP  #####\tTime (s): ', round(stop - start,2))
    plt.scatter(real_test, output, color="black")
    plt.show()

    # start = timeit.default_timer()
    # main_partial_BP()
    # output,result_online = test_BP_algorithm( trainset_size, s_min, s_max, on_test)
    # stop = timeit.default_timer()
    # reset_nn()
    # print('#####  PARTIAL BP  #####\tTime (s): ', round(stop - start,2))

    # dataset_X_train = dataset.xtable[:trainset_size]
    # dataset_X_test = dataset.xtable[trainset_size:]
    # dataset_y_train = dataset.ytable[:trainset_size]
    # dataset_y_test = dataset.ytable[trainset_size:]
    # regr = linear_model.LinearRegression()
    # regr.fit(dataset_X_train, dataset_y_train)
    # dataset_y_pred = regr.predict(dataset_X_test)
    # # print("Coefficients: \n", regr.coef_)
    # print("Mean squared error: %.2f" % mean_squared_error(dataset_y_test, dataset_y_pred))
    # print("Coefficient of determination: %.2f" % r2_score(dataset_y_test, dataset_y_pred))

    # x = np.array(dataset_X_test)
    # x = np.transpose(x)
    # x = x[0]
    # plt.scatter(x, dataset_y_test, color="black")
    # plt.scatter(x, dataset.ytable[trainset_size:], color="orange")
    # plt.plot(x, dataset_y_pred, color="blue", linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    # exit()

def read_params():
    global activation, epochs, alpha, n, alpha, nn, trainset_size_perc, dataset_name, s_max, s_min
    f = open('params.txt')
    params = f.readlines()
    params = [param.replace('\n','') for param in params]
    dataset_name = params[0]
    trainset_size_perc = int(params[1])
    activation = activations.index(params[2])
    nn.L = int(params[3])
    nn.n = [int(i) for i in params[4].split(' ')]
    epochs = int(params[5])
    n = float(params[6])
    alpha = float(params[7])
    s_min, s_max = [float(i) for i in params[8].split(' ')]
    

# layers = [[4,5,1],[4,5,6,1],[4,5,6,6,1],[4,5,6,6,5,1]]
# s_min = 0.1
# s_max = 0.9
# dataset_name = 'A1-turbine.txt'
# activation = 2 #index of activation function
# epochs = 500 #number of epochs
# n = 0.1 # learning rate [0.01 - 0.2]
# alpha = 0.9 # momentum [0.1 - 0.9]
# nn.L = 3 #number of layers
# nn.n = [4,5,1] #number of neural in each layer
# partial = 5 #number of partial batched set

activations = ['sigmoid','relu','tanh','linear']
nn = NN()
on_test = "online_test.csv"
activation = epochs = alpha = n = alpha = trainset_size_perc = dataset_name = s_max = s_min = 0
read_params()
dataset = read_dataset(dataset_name)
trainset_size_perc = 80
trainset_size = int(dataset.ns * trainset_size_perc / 100)
real_test = np.copy(dataset.ytable[trainset_size:])
scale_dataset(s_min,s_max)
main()
# run_all_params()
# find_best_params(train=True)
# order_best_params()
# plot_error('180_0')