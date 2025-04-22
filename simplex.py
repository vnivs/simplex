import numpy as np
from standardization_lp import standardization_lp

class simplex:
    def __init__(self, c, A, b,sense = "min", constraint_types = None, variable_type = None):
        self.c_std, self.A_std, self.b_std, self.basic_indices = standardization_lp(c, A, b, constraint_types, variable_type)
        self.m, self.n = self.A_std.shape
        self.max_iteration = 10000
        self.iteration = 0
        #构建单纯形表
        self.tableau = np.zeros((self.m +1, self.n +1))
        #把标准化后的系数矩阵，目标函数，RHS填充到单纯形表中，注意目标函数填入形式 z- c1x1 - c2x2 - ... - cncn = 0
        self.tableau[:-1 , :-1] = self.A_std
        self.tableau[:-1, -1] = self.b_std
        self.tableau[-1, :-1] = -self.c_std #检验数的负数
        #记录基变量和非基变量的位置
        self.basic = self.basic_indices.copy()
        self.non_basic = [i for i in range(self.n) if i not in self.basic]

        #判断是否达到最优---有无穷多最优解
        #如果如果非基变量的检验数都>=0且存在一个非基变量的检验数=0，则换入基中后，目标函数值不变，则两点连线上所有点都是最优解
    def _check_inf_sulotion(self):
        reduced_costs = -self.tableau[-1, :-1]
        for i in self.non_basic:
            if abs(reduced_costs[i]) < 1e-10:
                    return True
        return False

    #寻找入基变量
    def _find_entering_var(self):
        reduced_costs = -self.tableau[-1, :-1]
        candidates = np.where(reduced_costs > 0)[0]
        if len(candidates) == 0:
            return None #检验数都大于等于0，达到最优解
        return min(candidates) #若出现退化，防止进入cycling，使用bland规则

    #寻找出基变量
    def _find_leaving_var(self,entering_var):
        ratios = np.inf * np.ones(self.m)
        for i in range(self.m):
            if self.tableau[i,entering_var] > 0:
                ratios[i] = self.tableau[i, -1] / self.tableau[i, entering_var]

        min_ratio = min(ratios)
        if min_ratio == np.inf:
            return None  #具有无界解
        candidates = np.where(ratios == min_ratio)[0] #bland原则
        return min(candidates)

    #执行转轴操作
    def _pivot(self, entering_var, leave_var):
        #更新基变量和非基变量
        self.basic[leave_var] = entering_var
        self.non_basic = [i for i in range(self.n) if i not in self.basic]
        #获取主元和主元行
        pivot_row = self.tableau[leave_var].copy()
        pivot_element = pivot_row[entering_var]
        #主元行除主元
        pivot_row = pivot_row / pivot_element
        #高斯消元
        for i in range(self.m + 1):
            if i != leave_var:
                factor = self.tableau[i, entering_var]
                self.tableau[i] -= pivot_row * factor
        #更新主元行
        self.tableau[leave_var] = pivot_row

        #求解线性规划问题
    def solve(self):
        while self.iteration < self.max_iteration:
            self.iteration += 1
            entering_var = self._find_entering_var()
            if entering_var is None:
                #达到最优解
                solution = np.zeros(self.n)
                for i in range(self.m):
                    solution[self.basic[i]] = self.tableau[i, -1]
                #计算目标函数值
                obj_val = self.tableau[-1, -1]

                #检查是否有无穷多解
                if self._check_inf_sulotion():
                    return{
                        "status":"inf solutions",
                        "obj_val": obj_val
                    }
                else:
                    return{
                        "status":"unique solutions",
                        "solution": solution,
                        "obj_val": obj_val
                    }
            leave_var = self._find_leaving_var(entering_var)
            if leave_var is None:
                return{
                    "status":"inbounded solutions",
                }
            self._pivot(entering_var, leave_var)

        return{
            "status":"max_iterations",
        }


if __name__ == '__main__':
    # 示例1：有唯一最优解的问题
    print("\n=== 示例1：唯一最优解 ===")
    c = [3, 2, 1]
    A = [[1, 1, 1],
         [2, 1, 0],
         [1, 3, 2]]
    b = [10, 8, 12]
    solver = simplex(c, A, b, 'max')
    result = solver.solve()
    print(result)

    print("\n=== 示例2：无界解 ===")
    c_1 = [1, 1]
    A_1 = [[-1, -1]]
    b_1 = [-1]
    solver_1 = simplex(c_1, A_1, b_1, 'max')
    result_1 = solver_1.solve()
    print(result_1)

    print("\n=== 示例3：无穷多解 ===")
    c_2 = [1, 1]
    A_2 = [[1, 1]]
    b_2 = [1]
    solver_2 = simplex(c_2, A_2, b_2, 'max')
    result_2 = solver_2.solve()
    print(result_2)



