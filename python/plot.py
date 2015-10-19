import matplotlib.pyplot as plt

def plot_xy_projection(a, b):
    fig = plt.figure()
    x = a[:,0]
    y = a[:,1]
    c = x + y
    plt.scatter(x, y, c = c)
    x = b[:,0]
    y = b[:,1]
    plt.scatter(x,y,c = c)
    plt.show()


