import numpy as np
import math
import matplotlib.pyplot as plt
import line_search_methods
from matplotlib import rcParams
"""
x is the point to be calculated
f is the function to be optimized
step is the step size
"""
def calculate_analytical_gradient(x, f, d):#精确梯度
    #凸函数梯度
    # U = np.zeros(10)
    # for i in range(0, 3):
    #     U[i] = 4*x[i]**3+np.exp(x[i])
    # for i in range(3, 8):
    #     U[i] =2*x[i]/(x[i]**2 +1)+2*x[i]
    # for i in range(8, 10):
    #     U[i]=2*x[i]+np.cos(x[i])

    #多峰函数梯度
    dim = len(x)
    pi = np.pi
    t=gUfun(x, 10, 100, 4)
    U = np.zeros(dim)
    U[0]=(pi / dim) *(5*pi*np.sin(pi * (x[0] + 5) / 4)*np.cos(pi *  (x[0] + 5) / 4)+
            ((x[0] + 1) / 8)*(1+10*(np.sin(pi* (x[1] + 5) / 4))**2)) + t[0]
    for i in range(1, dim-1):
        U[i]=(pi / dim) *(((x[i] + 1) / 8)*(1+10*(np.sin(pi* (x[i+1] + 5) / 4))**2)+
                 ((x[i-1] + 1)**2 / 16)*5*pi*np.sin(pi * (x[i] + 5) / 4)*np.cos(pi * (x[i] + 5) / 4))+t[i]
    U[dim-1]=(pi / dim) *(((x[dim-2] + 1)**2 / 16)*5*pi*np.sin(pi *  (x[dim-1] + 5) / 4)*np.cos(pi *  (x[dim-1] + 5) / 4) +
                 (x[dim-1] + 1) /8 )+t[dim-1]

    return  U



def gUfun(x, a, k, m):
    dim = len(x)
    U = np.zeros(dim)
    for i in range(dim):
        if x[i] > a:
            U[i] = m*k * ((x[i] - a) ** (m-1))
        elif x[i] < -a:
            U[i] = -m*k * ((-x[i] - a) ** (m-1))
        else:
            U[i] = 0
    return U

def calculate_one_ways_differential_gradient(x, f, d):
    list = []
    for i in range(0, x.size):
        x[i] += d
        y1 = f(x)
        x[i] -= d
        y2 = f(x)
        value = (y1 - y2) / d
        list.append(value)
    return np.array(list)

def calculate_double_way_differential_gradient(x, f, d):
    list = []
    for i in range(0, x.size):
        x[i] += d
        y1 = f(x)
        x[i] -= (2*d)
        y2 = f(x)
        x[i] += d
        value = y1 - y2
        value /= (2*d)
        list.append(value)
    return np.array(list)

def draw_one_way_differential_gradient(f, x0): #画数值梯度与精确梯度误差图
    x = x0
    x_label = []
    y_label = []
    for i in range(1, 20):
        val = np.linalg.norm(calculate_one_ways_differential_gradient(x, f, 10**(-i)) - calculate_analytical_gradient(x, f, 10**(-i)),2)
        val /= np.linalg.norm(calculate_analytical_gradient(x, f, 10**(-i)),2)
        x_label.append(-i)
        y_label.append(math.log(val,10))
    print(x_label)
    print(y_label)
    plt.rc('text', usetex=True)

    plt.plot(x_label, y_label)
    plt.xlabel(r'$\lg\epsilon $',
           labelpad=-10,  #调整x轴标签与x轴距离
           x=1.,)
    plt.ylabel(r'$\lg(\frac{\Vert \nabla f-\nabla \overline f \Vert}{\Vert\nabla f\Vert} ) $',
           labelpad=-10,  #调整y轴标签与y轴的距离
           y=1.,rotation=0,)
    plt.gca().invert_xaxis()
    plt.show()