import numpy as np

def standardization_lp(c, A ,b, sense = "min", constraint_types = None, variable_type = None, M = 1e6):
    """
    :param c: 价值向量
    :param A: 系数矩阵
    :param b: 资源向量
    :param sense: 目标函数类型 min or max
    :param constraint_types: 约束类型 >= <= =
    :param variable_type: 决策变量符号 >=0 <= 0 free(无约束)
    :param M  使用大M法时引入的一个充分大的正数，可以根据具体问题的情况进行调节
    """
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)

    #设置默认参数
    if constraint_types is None:
        constraint_types = ["<="] * len(b)
    if variable_type is None:
        variable_type = [">=0"] * len(c)

    #如果目标函数为最大化，则价值向量取负号
    if sense == "max":
        c = -c

    #1.先处理决策变量
    orignal_var = len(c)
    extra_var = 0
    for type in variable_type:
        if type == "free":
            extra_var += 1

    new_c = np.zeros(orignal_var + extra_var)
    new_A = np.zeros((len(b), orignal_var + extra_var))

    extra_var_pos = orignal_var
    for i,type in enumerate(variable_type):
        if type == ">=0": #不处理和原始保持一致
            new_c[i] = c[i]
            new_A[:,i] = A[:,i]
        elif type == "<=0": #添加一个负号
            new_c[i] = -c[i]
            new_A[:,i] = -A[:,i]
        else: #如果x is free 则用 x = x' - x'' 其中x', x''>=0
            new_c[i] = c[i]
            new_c[extra_var_pos] = -c[i]
            new_A[:,i] = A[:,i]
            new_A[:,extra_var_pos] = -A[:,i]
            extra_var_pos += 1

    # 2. 处理约束条件
    #记录需要添加的变量的个数，包括松弛变量(slack variable)、剩余变量(surplus variable)、人工变量(artificial variable)
    slack_count = 0
    surplus_count = 0
    artificial_count = 0
    for type in constraint_types:
        if type == "<=":
            slack_count += 1
        elif type == ">=":
            surplus_count += 1
            artificial_count += 1
        else:
            artificial_count += 1

    total_var = orignal_var +extra_var + slack_count + surplus_count + artificial_count
    final_c = np.zeros(total_var)
    final_A = np.zeros((len(b), total_var))

    #复制已经有的系数矩阵和价值向量
    final_c[:(orignal_var + extra_var)] = new_c
    final_A[:,:(orignal_var + extra_var)] = new_A

    #为了确保剩余变量和人工变量在系数矩阵的最后，需要定义一下添加变量的位置
    # 变量顺序：原始变量->自由变量转换后添加的变量->剩余变量->松弛变量+人工变量
    surplus_start = orignal_var + extra_var
    basis_start = surplus_start + surplus_count

    for i,type in enumerate(constraint_types):
        if type == "<=":
            final_A[i,basis_start] = 1
            basis_start += 1
        elif type == ">=":
            final_A[i,surplus_start] = -1
            final_A[i,basis_start] = 1
            final_c[basis_start] = M
            surplus_start += 1
            basis_start += 1
        else:
            final_a[i,basis_start] = 1
            final_c[basis_start] = M
            basis_start += 1
    final_variable_type = [">=0"] * total_var
    final_constraint_type = ["="] * len(b)
    return final_c, final_A, b, final_variable_type, final_constraint_type


if __name__ == '__main__':
    # 一个例子：
    # 最大化：3x1 + 2x2 + 3x3
    # 约束条件：
    # x1 + x2 + x3 <= 4
    # 2x1 + x2 + 3x3 >= 3
    # 3x1 + 4x2 + 5x3 = 5
    # x1 >= 0, x2 是自由变量, x3 <= 0

    # 定义问题参数
    c = [3, 2, 3]
    A = [[1, 1, 1],
         [2, 1, 3],
         [3, 4, 5]]
    b = [4, 3, 5]
    sense = 'max'
    constraint_types = ['>=', '>=', '>=']
    variable_types = ['>=0', 'free', '<=0']

    # 转换为标准型
    c_std, A_std, b_std, _, _ = standardization_lp(c, A, b, sense, constraint_types, variable_types)
    print("\n最终结果:")
    print("转换后的目标函数系数：", c_std)
    print("转换后的约束矩阵：")
    print(A_std)
    print("约束右侧值：", b_std)