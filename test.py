import numpy as np
from constrained_optimization import QuadraticPenaltyMethod
import matplotlib.pyplot as plt

# 使用beale函数作为目标函数
def objective_function(x):
    #return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2
    return ((1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)) * (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)))

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

def inequality_constraint1(x):
    # x小于等于4.5
    return x[0] - 2
def inequality_constraint2(x):
    # y小于等于4.5
    return x[1] - 2
def inequality_constraint3(x):
    # x大于等于-4.5
    return -x[0] - 2
def inequality_constraint4(x):
    # y大于等于-4.5
    return -x[1] - 2

# 定义等式约束（例如：前两个变量相等）
def equality_constraint(x):
    return x[0] - x[1]

# 初始点（30维）
x0 = np.array([1.0] * 2)

# 创建优化器
optimizer = QuadraticPenaltyMethod(
    objective_func=objective_function,
    inequality_constraints=[inequality_constraint1, inequality_constraint2, inequality_constraint3, inequality_constraint4],
    #equality_constraints=[equality_constraint],
    mu=1.0,
    beta=10.0,
    epsilon=1e-6
)

# 优化
x_opt, f_opt = optimizer.optimize(x0)

# 输出结果
print(f"最优解: {x_opt}")
print(f"最优值: {f_opt}")
print(f"不等式约束1违反程度: {inequality_constraint1(x_opt)}")
print(f"不等式约束2违反程度: {inequality_constraint2(x_opt)}")
print(f"等式约束违反程度: {equality_constraint(x_opt)}")