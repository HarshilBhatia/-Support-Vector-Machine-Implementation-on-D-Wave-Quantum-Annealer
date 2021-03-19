import neal
import math
from dwave.system import LeapHybridSampler
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import time

TYPE = 'SA'
data_file = 'banknote_1'

# Select the solver
if(TYPE == 'HQPU'):
    sampler = LeapHybridSampler()
if(TYPE == 'SA'):
    sampler = neal.SimulatedAnnealingSampler()
if(TYPE == 'QPU'):
    sampler = EmbeddingComposite(DWaveSampler())


N = 20
validation_pts = 10
xi = 0.001


def kernel(x, y, gamma):
    if gamma == -1:
        k = np.dot(x, y)
    elif gamma >= 0:
        k = np.exp(-gamma*(np.linalg.norm(x-y, ord=2)))
    return k

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def predict_class(x_test, alpha, b):
    N = len(alpha)
    f = sum([alpha[n]*t[n]*kernel(data[n], x_test, gamma)
                for n in range(N)]) + b
    return f

def train_SVM(x, K, t):

    Q_tilde = np.zeros((K*N, K*N))
    for n in range(N):
        for m in range(N):
            for k in range(K):
                for j in range(K):
                    Q_tilde[(K*n+k, K*m+j)] = 0.5*(B**(k+j))*t[n]*t[m] * \
                        (kernel(x[n], x[m], gamma)+xi) - \
                        (delta(n, m)*delta(k, j)*(B**k))

    Q = np.zeros((K*N, K*N))
    for j in range(K*N):
        Q[(j, j)] = Q_tilde[(j, j)]
        for i in range(K*N):
            if i < j:
                Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

    size_of_q = Q.shape[0]
    qubo = {(i, j): Q[i, j]
            for i, j in product(range(size_of_q), range(size_of_q))}

    now = time.time()
    if(TYPE == 'HQPU'):
        response = sampler.sample_qubo(qubo)
    if(TYPE == 'SA'):
        response = sampler.sample_qubo(qubo, num_reads=100)
    if(TYPE == 'QPU'):
        response = sampler.sample_qubo(qubo, num_reads=100)

    print(time.time() - now) 

    a = response.first.sample
    print(response.first.energy)

    alpha = {}
    for n in range(N):
        alpha[n] = sum([(B**k)*a[K*n+k] for k in range(K)])

    b = sum([alpha[n]*(C-alpha[n])*(t[n]-(sum([alpha[m]*t[m]*kernel(x[m], x[n], gamma)
                                                for m in range(N)]))) for n in range(N)])/sum([alpha[n]*(C-alpha[n]) for n in range(N)])

    return alpha, b

# ----- Predict class of a data point -----

# ----- Set up parameters -----
B = 2  # base used for encoding the real variables
K = 2  # number of binary variables used for encoding
C = 3  # regularization parameter

# constraint penalty
gamma = 16  # kernel hyperparameter

#load the data file
training_data = np.loadtxt('./data/{}.txt'.format(data_file), delimiter=',')\

for i in range(N+validation_pts):
    if(training_data[i][-1] == 0):
        training_data[i][-1] = -1

data = training_data[:N+validation_pts, :2]
t = training_data[:N + validation_pts, -1]

x_min, x_max = 1000, 0
y_min, y_max = 1000, 0
# rescalling data
for i in range(N+validation_pts):
    x_min = min(data[i][0], x_min)
    x_max = max(data[i][0], x_max)
    y_min = min(data[i][1], y_min)
    y_max = max(data[i][1], y_max)

for i in range(N+validation_pts):
    data[i][0] = (data[i][0] - x_min)/(x_max - x_min)
    data[i][1] = (data[i][1] - y_min)/(y_max - y_min)
alpha, b = train_SVM(data, K, t)

# Plot results:
plot_fig = 0

if plot_fig:
    plt.figure()
    cm = plt.cm.RdBu

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80),
                            np.linspace(0.0, 1.0, 80))
    Z = []
    for row in range(len(xx)):
        Z_row = []
        for col in range(len(xx[row])):
            target = np.array([xx[row][col], yy[row][col]])
            Z_row.append(predict_class(target, alpha, b))
        Z.append(Z_row)

    cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1,
                                                    0.1), cmap=cm, alpha=0.8, extend="both")

    plt.contour(xx, yy, Z, levels=[0.0], colors=(
        "black",), linestyles=("--",), linewidths=(0.8,))
    plt.colorbar(cnt, ticks=[-1, 0, 1])

    red_sv = []
    blue_sv = []
    red_pts = []
    blue_pts = []

    for i in range(N):
        if(alpha[i]):
            if(t[i] == 1):
                blue_sv.append(data[i, :2])
            else:
                red_sv.append(data[i, :2])

        else:
            if(t[i] == 1):
                blue_pts.append(data[i, :2])
            else:
                red_pts.append(data[i, :2])

    plt.scatter([el[0] for el in blue_sv],
                [el[1] for el in blue_sv], color='b', marker='^', edgecolors='k', label="Type 1 SV")

    plt.scatter([el[0] for el in red_sv],
                [el[1] for el in red_sv], color='r', marker='^', edgecolors='k', label="Type -1 SV")

    plt.scatter([el[0] for el in blue_pts],
                [el[1] for el in blue_pts], color='b', marker='o', edgecolors='k', label="Type 1 Train")

    plt.scatter([el[0] for el in red_pts],
                [el[1] for el in red_pts], color='r', marker='o', edgecolors='k', label="Type -1 Train")


tp, fp, tn, fn = 0, 0, 0, 0
for i in range(N, N+validation_pts):
    cls = predict_class(data[i], alpha, b)
    y_i = t[i]
    if(y_i == 1):
        if(cls > 0):
            tp += 1
        else:
            fp += 1
    else:
        if(cls < 0):
            tn += 1
        else:
            fn += 1

# calculate KPI's

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_score = tp/(tp + 1/2*(fp+fn))
accuracy = (tp + tn)/(tp+tn+fp+fn)

print("f1_score = {} accuracy = {} precision = {} recall = {}".format(f_score, accuracy,precision,recall))

if plot_fig:
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('./{}/{}_{}'.format(TYPE,data_file,N))
    # plt.savefig("SA.png")

