for i, ele in enumerate(x_train):
    a=np.mean(ele)
    b=np.std(ele)
    for j, elem in enumerate(ele):
        x_train[i][j] = (elem-a)/b


for i, ele in enumerate(x_train):
    for j, elem in enumerate(ele):
        x_train[i][j] += np.random.normal(0, 0.2)
