"""
kmeans:
无监督学习，分类目标未知
优点:

属于无监督学习，无须准备训练集
原理简单，实现起来较为容易
结果可解释性较好
缺点:

需手动设置k值。 在算法开始预测之前，我们需要手动设置k值，即估计数据大概的类别个数，不合理的k值会使结果缺乏解释性
可能收敛到局部最小值, 在大规模数据集上收敛较慢
对于异常点、离群点敏感

：pram_1 簇
：pram_2 质心

"""
#首先，构建随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1] # 列的数量，即数据的特征个数
    centroids = mat(zeros((k,n))) # 创建k个质心矩阵
    for j in range(n): # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j])    # 最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)    # 范围 = 最大值 - 最小值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))    # 随机生成，mat为numpy函数，需要在最开始写上 from numpy import *
    return centroids
#分配质心，并重新计算新质心
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    # 行数，即数据个数
    clusterAssment = mat(zeros((m, 2)))    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataSet, k)    # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 循环每一个数据点并分配到最近的质心中去
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                clusterAssment[i, :] = minIndex,minDist**2    # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print centroids
        for cent in range(k): # 更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取该簇中的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment

#二分聚类算法
"""
将所有点看成一个簇
当簇数目小于 k 时
对于每一个簇
计算总误差
在给定的簇上面进行 KMeans 聚类（k=2）
计算将该簇一分为二之后的总误差
选择使得误差最小的那个簇进行划分操作

"""