import numpy as np
from line_search_methods import grad_strongwolfe
from get_gradient import calculate_double_way_differential_gradient

class QuadraticPenaltyMethod:
    def __init__(self, objective_func, inequality_constraints=None, equality_constraints=None, 
                mu=1.0, beta=10.0, epsilon=1e-6):
        self.f = objective_func
        self.inequality_constraints = inequality_constraints if inequality_constraints else []
        self.equality_constraints = equality_constraints if equality_constraints else []
        self.mu = mu
        self.beta = beta
        self.epsilon = epsilon
        self.tao = 0.5
        self.rou = 0.1

    def penalty_function(self, x):
        """Compute the quadratic penalty function with both equality and inequality constraints"""
        penalty = 0.0
        # 处理不等式约束 g(x) <= 0
        for g in self.inequality_constraints:
            constraint_val = g(x)
            penalty += max(0, constraint_val)**2
        
        # 处理等式约束 h(x) = 0
        for h in self.equality_constraints:
            constraint_val = h(x)
            penalty += constraint_val**2
            
        return self.f(x) + self.mu * penalty

    def optimize(self, x0, max_iter=10000):
        x = x0.copy()
        y = []
        prev_penalty = float('inf')
        y.append(self.penalty_function(x0))
        
        for k in range(max_iter):
            def gradient_func(x, f, step):
                return calculate_double_way_differential_gradient(x, f, step)
            
            x, val, _, _ = grad_strongwolfe(self.penalty_function, gradient_func, x, 1e-4, self.epsilon)
            current_penalty = self.penalty_function(x)
            y.append(current_penalty)
            
            # 检查约束违反程度
            inequality_violation = max([max(0, g(x)) for g in self.inequality_constraints], default=0)
            equality_violation = max([abs(h(x)) for h in self.equality_constraints], default=0)
            max_violation = max(inequality_violation, equality_violation)
            
            # 计算惩罚值的相对变化
            penalty_change = abs(prev_penalty - current_penalty) / (abs(prev_penalty) + 1e-10)
            
            # 动态调整 beta
            if max_violation > 0.1:  # 约束违反严重
                self.beta = min(10.0, self.beta * 2)  # 快速增加惩罚
            elif penalty_change < 1e-3:  # 收敛较慢
                self.beta = max(1.1, self.beta * 0.8)  # 减小惩罚增长率
            else:  # 正常情况
                self.beta = 1.5  # 使用默认值
            
            if max_violation < self.epsilon:
                break
            
            self.mu *= self.beta
            prev_penalty = current_penalty
            
        return x, self.f(x), y