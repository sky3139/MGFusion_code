# -*- coding: utf-8 -*-
# @Time : DATE:2021/8/29
# @Author : yan
# @Email : 1792659158@qq.com
# @File : blogDemo.py

import matplotlib.pyplot as plt
import numpy as np


def generate_rubik_cube(nx, ny, nz):
    """
    根据输入生成指定尺寸的魔方
    :param nx:
    :param ny:
    :param nz:
    :return:
    """
    # 准备一些坐标
    n_voxels = np.ones((nx + 2, ny + 2, nz + 2), dtype=bool)

    # 生成间隙
    size = np.array(n_voxels.shape) * 2
    filled_2 = np.zeros(size - 1, dtype=n_voxels.dtype)
    filled_2[::2, ::2, ::2] = n_voxels

    # 缩小间隙
    # 构建voxels顶点控制网格
    # x, y, z均为6x6x8的矩阵，为voxels的网格，3x3x4个小方块，共有6x6x8个顶点。
    # 这里//2是精髓，把索引范围从[0 1 2 3 4 5]转换为[0 0 1 1 2 2],这样就可以单独设立每个方块的顶点范围
    x, y, z = np.indices(np.array(filled_2.shape) +
                         1).astype(float) // 2  # 3x6x6x8，其中x,y,z均为6x6x8

    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # 修改最外面的面
    x[0, :, :] += 0.94
    y[:, 0, :] += 0.94
    z[:, :, 0] += 0.94

    x[-1, :, :] -= 0.94
    y[:, -1, :] -= 0.94
    z[:, :, -1] -= 0.94

    # 去除边角料
    filled_2[0, 0, :] = 0
    filled_2[0, -1, :] = 0
    filled_2[-1, 0, :] = 0
    filled_2[-1, -1, :] = 0

    filled_2[:, 0, 0] = 0
    filled_2[:, 0, -1] = 0
    filled_2[:, -1, 0] = 0
    filled_2[:, -1, -1] = 0

    filled_2[0, :, 0] = 0
    filled_2[0, :, -1] = 0
    filled_2[-1, :, 0] = 0
    filled_2[-1, :, -1] = 0

    # 给魔方六个面赋予不同的颜色
    colors = np.array(['#ffd400', "#fffffb", "#f47920",
                      "#d71345", "#145b7d", "#45b97c"])
    facecolors = np.full(filled_2.shape, '#77787b')  # 设一个灰色的基调
    # facecolors = np.zeros(filled_2.shape, dtype='U7')
    facecolors[-1:, -1:, -1] = colors[0]  # 上黄
    # facecolors[:, :, 0] = colors[1]	    # 下白
    # facecolors[:, 0, :] = colors[2]  	# 左橙
    # facecolors[:, -1, :] = colors[3]	# 右红
    # facecolors[0, :, :] = colors[4]	    # 前蓝
    # facecolors[-1, :, :] = colors[5]	# 后绿

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=facecolors)
    plt.show()


# if __name__ == '__main__':
#     generate_rubik_cube(3, 3, 3)
def draw_voxel():
    import matplotlib.pyplot as plt
    import numpy as np
    # 准备一组体素坐标
    n_voxels = np.ones((3, 3, 3), dtype=bool)

    # x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2

    # x[1::2, :, :] += 0.95
    # y[:, 1::2, :] += 0.95
    # z[:, :, 1::2] += 0.95

    size = np.array(n_voxels.shape) * 2
    filled_2 = np.zeros(size - 1, dtype=n_voxels.dtype)
    filled_2[::2, ::2, ::2] = n_voxels

    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    colors = np.array(['#ff0000', "#fffffb", "#f47920",
                      "#d71345", "#145b7d", "#45b97c"])
    facecolors = np.full(filled_2.shape, '#77787b')  # 设一个灰色的基调
    # facecolors = np.zeros(filled_2.shape, dtype='U7')
    facecolors[3:, :2, -1] = colors[0]  # 上黄p
    x[1::2, :, :] += 0.94
    y[:, 1::2, :] += 0.94
    z[:, :, 1::2] += 0.94
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=facecolors)
    # plt.axis('off')
    plt.show()


def draw_cube():
    import matplotlib.pyplot as plt
    import numpy as np
    # 准备一组体素坐标
    n_voxels = np.ones((6, 6, 6), dtype=bool)

    # x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2

    # x[1::2, :, :] += 0.95
    # y[:, 1::2, :] += 0.95
    # z[:, :, 1::2] += 0.95

    size = np.array(n_voxels.shape) * 2
    filled_2 = np.zeros(size - 1, dtype=n_voxels.dtype)
    filled_2[::2, ::2, ::2] = n_voxels

    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    colors = np.array(['#0000ff', "#fffffb", "#f47920",
                      "#d71345", "#145b7d", "#45b97c"])
    facecolors = np.full(filled_2.shape, '#77787b')  # 设一个灰色的基调
    # facecolors = np.zeros(filled_2.shape, dtype='U7')
    # facecolors[:, :,:] = colors[0]  # 上黄p
    x[1::2, :, :] += 0.94
    y[:, 1::2, :] += 0.94
    z[:, :, 1::2] += 0.94
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=facecolors)
    plt.axis('off')
    plt.show()


draw_cube()
