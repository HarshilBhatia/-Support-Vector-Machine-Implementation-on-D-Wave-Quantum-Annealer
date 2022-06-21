
import numpy as np



def load_data(data_file,N,validation_pts):
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

    return data,t