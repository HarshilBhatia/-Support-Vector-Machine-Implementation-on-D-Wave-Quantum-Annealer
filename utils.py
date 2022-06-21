
import matplotlib.pyplot as plt
import numpy as np

def  plot_figure(SVM,alpha,data,t,b,N,sampler_type):
    plt.figure()
    cm = plt.cm.RdBu

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80),
                            np.linspace(0.0, 1.0, 80))
    Z = []
    for row in range(len(xx)):
        Z_row = []
        for col in range(len(xx[row])):
            target = np.array([xx[row][col], yy[row][col]])
            Z_row.append(SVM.predict_class(target, alpha, b))
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

    
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig(f'{sampler_type}.jpg')

def compute_metrics(SVM,alpha,data,t,b,N,validation_pts):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(N, N+validation_pts):
        predicted_cls = SVM.predict_class(data[i], alpha, b)
        y_i = t[i]
        if(y_i == 1):
            if(predicted_cls > 0):
                tp += 1
            else:
                fp += 1
        else:
            if(predicted_cls < 0):
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = tp/(tp + 1/2*(fp+fn))
    accuracy = (tp + tn)/(tp+tn+fp+fn)

    return precision,recall,f_score,accuracy


