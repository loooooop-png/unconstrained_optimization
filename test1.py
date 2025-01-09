import numpy as np
from constrained_optimization import QuadraticPenaltyMethod
import matplotlib.pyplot as plt
import math

# 使用beale函数作为目标函数
def objective_function(x):
    #return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2
    #return ((1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)) * (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)))
    #return (x[0] - 1)**2 + (x[1] - 2)**2
    return (x[0] - 1)**2 + (x[0] - x[1])**2 + (x[1] - x[2])**2
# # 定义约束条件
# def inequality_constraint1(x):
#     # x小于等于4.5
#     return x[0] - 4.5
# def inequality_constraint2(x):
#     # y小于等于4.5
#     return x[1] - 4.5
# def inequality_constraint3(x):
#     # x大于等于-4.5
#     return -x[0] - 4.5
# def inequality_constraint4(x):
#     # y大于等于-4.5
#     return -x[1] - 4.5


# def inequality_constraint1(x):
#     # x小于等于4.5
#     return x[0] - 2
# def inequality_constraint2(x):
#     # y小于等于4.5
#     return x[1] - 2
# def inequality_constraint3(x):
#     # x大于等于-4.5
#     return -x[0] - 2
# def inequality_constraint4(x):
#     # y大于等于-4.5
#     return -x[1] - 2
'''
# 定义不等式约束
def inequality_constraint1(x):
    # x小于等于3
    return x[0] - 3
def inequality_constraint2(x):
    # y小于等于4
    return x[1] - 4
def inequality_constraint3(x):
    # x大于等于0
    return -x[0] 
def inequality_constraint4(x):
    # y大于等于1
    return -x[1] + 1
# 定义等式约束
def equality_constraint(x):
    return 2*x[0] + 3*x[1] - 5
'''
# 定义不等式约束
def inequality_constraint1(x):
    # x小于等于10
    return x[0] - 10
def inequality_constraint2(x):
    # y小于等于10
    return x[1] - 10
def inequality_constraint3(x):
    # z小于等于10
    return x[2] - 10 
def inequality_constraint4(x):
    # x大于等于-10
    return -x[0] - 10
def inequality_constraint5(x):
    # y大于等于-10
    return -x[1] - 10
def inequality_constraint6(x):
    # z大于等于-10
    return -x[2] - 10
# 定义等式约束
def equality_constraint(x):
    return x[0] * (1 + x[1]**2) + x[2]**4 - 4 - 3 * math.sqrt(2)

# 初始点（3维）
x0 = np.array([1.0] * 3)

# 创建优化器
optimizer = QuadraticPenaltyMethod(
    objective_func=objective_function,
    inequality_constraints=[inequality_constraint1, inequality_constraint2, inequality_constraint3, inequality_constraint4],
    equality_constraints=[equality_constraint],
    mu=1.0,
    beta=1.5,
    epsilon=1e-6
)

# 优化
x_opt, f_opt, y = optimizer.optimize(x0)

# 输出结果
print(f"最优解: {x_opt}")
print(f"最优值: {f_opt}")
print(f"不等式约束1违反程度: {inequality_constraint1(x_opt)}")
print(f"不等式约束2违反程度: {inequality_constraint2(x_opt)}")
print(f"等式约束违反程度: {equality_constraint(x_opt)}")
iterations = range(len(y))
plt.plot(iterations, y)
plt.xlabel('Iteration')
plt.ylabel('Penalty Function Value')
plt.title('Gradient Descent Convergence Curve')
plt.show()