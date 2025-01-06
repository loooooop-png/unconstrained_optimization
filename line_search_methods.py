import numpy as np
import math
import matplotlib.pyplot as plt
"""
fun is the function to be optimized
gfun is the gradient of fun
x0 is the initial point
step is the step size
alpha is the step size factor
c is the constant in the Wolfe condition
t is the constant in the backtracking method
"""
def grad_constant_step(fun, gfun, x0, alpha, step, e):
    maxk =5000;k = 0;epsilon = e;u=[fun(x0)];
    while k < maxk:
        gk = gfun(x0, fun, step)
        dk = -gk
        if np.linalg.norm(dk) < epsilon:
            break
        x0 = x0 + np.multiply(dk, alpha)
        u.append(fun(x0))
        k += 1
    x = x0
    val = fun(x)
    return x, val, k,u

def grad_trackbacking(fun, gfun, x0, alpha_, c, t, step, e):
    maxk = 5000;k = 0;epsilon = e;ministep = 1e-5
    flag = False;u=[fun(x0)];
    while k < maxk:
        gk = gfun(x0, fun, step)
        dk = -gk
        alpha = alpha_
        if np.linalg.norm(dk) < epsilon:
            break
        while fun(x0 + alpha * dk) > fun(x0) + c * alpha * gfun(x0, fun, step).T @ dk:
            if np.linalg.norm(alpha * dk) <= ministep:
                flag = True
                break
            alpha = alpha * t
        if flag:
            break
        x0 = x0 + alpha * dk
        u.append(fun(x0))
        k += 1
    x = x0
    val = fun(x)
    return x, val, k,u

def zoom(fun,gfun,x0,dk,alpha_lo,alpha_hi,step):
    c1 = 1e-4;c2 = 0.9;f0 = fun(x0);
    f1 = gfun(x0 , fun , step).T @ dk
    for i in range(1,50):
        flo = fun(x0 + alpha_lo * dk)
        alpha_j=0.5 * (alpha_lo + alpha_hi)
        f2 = fun(x0 + alpha_j * dk)
        if f2> f0 + c1 * alpha_j * f1 or f2 >= flo:
            alpha_hi = alpha_j
        else:
            f3 = gfun(x0 + alpha_j * dk, fun, step).T @ dk
            if np.linalg.norm(f3) <= -c2 * f1:
                return alpha_j
            if f3*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
    return 0.5*(alpha_lo + alpha_hi)

def grad_strongwolfe(fun, gfun, x0, step, e): #strongwolfe条件
    maxk = 5000;k = 0;epsilon = e;c1 = 1e-4;c2 = 0.9;alpha_max = 50; a = 1;
    fk_prev = gfun(x0, fun ,step).T @ -gfun(x0, fun, step)
    u=[fun(x0)];
    while k < maxk:
        dk = -gfun(x0, fun, step)
        if np.linalg.norm(dk) < epsilon:
            break
        alpha_prev = 0
        f_zero = fun(x0)
        f_prev = fun(x0)
        fg_zero = gfun(x0, fun ,step).T @ dk
        alpha = a*fk_prev/fg_zero
        for i in range(1,50):
            f0 = fun(x0 + alpha * dk)
            if (f0 > f_zero + c1 * alpha * fg_zero) or (f0 >= f_prev and i>1):
                alpha = zoom(fun,gfun,x0,dk,alpha_prev,alpha,step)
                break
            f1 = gfun(x0 + alpha * dk, fun, step).T @ dk
            if np.linalg.norm(f1) <= -c2 * fg_zero:
                break
            if f1 >= 0:
                alpha = zoom(fun,gfun,x0,dk,alpha,alpha_prev,step)
                break
            alpha_prev = alpha
            f_prev = f0
            alpha = 0.5 * (alpha +alpha_max)
        a = alpha
        fk_prev = fg_zero
        x0 = x0 + alpha * dk
        u.append(fun(x0))
        k += 1
    x = x0
    val = fun(x)
    return x, val, k,u