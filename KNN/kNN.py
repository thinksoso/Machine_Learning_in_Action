import numpy as np
import operator

#生成测试数据
def createDataset():
    data = [[1.1,1.0],[1.0,1.0],[0,0],[0,0.1]]
    group = np.array(data)
    labels = ['A','B','C','D']
    return group,labels

#kNN算法   计算要分类的向量与各向量之间的距离，找出距离最近的前k个，然后把前k个中出现次数最多
#的label作为要分类向量的label
def kNN_model_0(aim,train_data,train_label,k):
    ''' 
    aim是要分类的变量
    train_data是训练集 train_label是训练集的标签 两者行数一致，构成一个样本 
    k是评判标准
    距离采用欧式距离'''

    train_data_size = train_data.shape[0]
    diff = np.tile(aim,(train_data_size,1))-train_data
    distance = diff**2
    #一行的每一列相加
    distance = distance.sum(axis=1)
    #将distance的距离按从小到大排序后返回index的列表
    sorted_distance = distance.argsort()
    times = {}
    for i in range(int(k)):
        label = train_label[sorted_distance[i]]
        #对应标签出现次数加一，get不到default值为0
        times[label] = times.get(label,0)+1
    #下面对times字典进行排序,找出出现次数最多的label
    flag = 1
    result = ''
    for key in times:
        if flag == 1:
            result = key
            flag = 2
        else:
            if times[result] < times[key]:
                result=key
    return result


if __name__ == "__main__":
    group,label=createDataset()
    result = kNN_model_0([0,0],group,label,2)
    print (result)
            