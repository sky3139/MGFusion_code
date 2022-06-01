import numpy as np
import struct
 
TSDFfilename='chess.raw' # TSDF 二进制文件路径
PointCloudfilename="chessPointCloud.xyz" # 输出点云文件
h,w,d=512,512,512 # TSDF 的高度 宽度和深度
 
# 加载测试数据
f = open(TSDFfilename,'rb')
# 文档中包含的数字个数，而一个short数占2个字节
data_raw = struct.unpack('h'*(h*w*d),f.read(2*h*w*d)) # h: short H: usight short
f.close()
verify_data =  np.asarray(data_raw).reshape(h,w,d)
 
pointCloud = []
f_pc = open(PointCloudfilename, "w")
 
# 转换为点云的遍历过程
for i in range(h):
    for j in range(w):
        for k in range(1,d):
            if (verify_data[i][j][k]>0 and verify_data[i][j][k-1]<0):
                # print(i,j,k)
                pointCloud.append([i,j,k])
                f_pc.write(str(i)+" "+str(j)+" "+str(k)+" \n")
 
for i in range(h):
    for k in range(d):
        for j in range(1,w):
            if (verify_data[i][j][k]>0 and verify_data[i][j-1][k]<0):
                # print(i,j,k)
                if [i,j,k] in pointCloud: # 避免记录同样的点
                    pass
                else:
                    f_pc.write(str(i)+" "+str(j)+" "+str(k)+" \n")
 
for k in range(d):
    for j in range(w):
        for i in range(1,h):
            if (verify_data[i][j][k]>0 and verify_data[i-1][j][k]<0):
                # print(i,j,k)
                if [i,j,k] in pointCloud: # 避免记录同样的点
                    pass
                else:
                    f_pc.write(str(i)+" "+str(j)+" "+str(k)+" \n")
f_pc.close()
 
#下面部分的代码可以单独拿出去用作可视化，改一下文件名那里的参数就可以了
import open3d as o3d
 
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(PointCloudfilename)#可以替换成需要查看的点云文件
o3d.visualization.draw_geometries([pcd])