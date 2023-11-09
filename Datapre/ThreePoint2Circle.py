###########################################################################
# 由圆上三点确定圆心和半径
###########################################################################
# INPUT
# p1   :  - 第一个点坐标, list或者array 1x3
# p2   :  - 第二个点坐标, list或者array 1x3
# p3   :  - 第三个点坐标, list或者array 1x3
# 若输入1x2的行向量, 末位自动补0, 变为1x3的行向量
###########################################################################
# OUTPUT
# pc   :  - 圆心坐标, array 1x3
# r    :  - 半径, 标量
###########################################################################
# 调用示例1 - 平面上三个点
# pc1, r1 = points2circle([1, 2], [-2, 1], [0, -3])
# 调用示例2 - 空间中三个点
# pc2, r2 = points2circle([1, 2, -1], [-2, 1, 2], [0, -3, -3])
###########################################################################
import numpy as np
from numpy.linalg import det

def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)

    # 输入检查
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None

    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10**-6:
        print('\t三点共线, 无法确定圆')
        return None

    temp1 = np.vstack((p1, p2, p3))
    temp2 = np.ones(3).reshape(3, 1)
    mat1 = np.hstack((temp1, temp2))  # size = 3x4

    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)

    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))  # size = 4x5

    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])

    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)

    return pc
