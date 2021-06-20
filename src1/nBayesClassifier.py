import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}
    '''
    通过一个数据子集去计算均值和标准差
    data_subset是一个数组（array）类型数据
    '''
    def mean_and_standard_deviation(self,data_subset):
        mean = np.average(data_subset)
        s_d = np.sqrt(np.var(data_subset))
        return (mean,s_d)
    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        # 对于连续的数据，我们采用第二种方式（使用高斯分布拟合的方法）来实现
        
        # 先统计c1，c2，c3的概率
        # 遍历所有训练数据
        # 我们需要统计
        '''
            1:number of category
            2:建立subset array with a certain category and a certain feature
            3:for discrete feature[0],我们应该统计pxc
        '''
        num_c = {1:0,2:0,3:0}
        num_c_feature = {(1,1):0,(1,2):0,(1,3):0,(2,1):0,(2,2):0,(2,3):0,(3,1):0,(3,2):0,(3,3):0}
        subset_array_dict = {}
        subset_array_cnt = {}
        for i in range(1,4):
              for j in range(1,8):
                  subset_array_cnt[(i,j)] = 0
        for i in range(traindata.shape[0]):
            num_c[int(trainlabel[i])] += 1
            #更新相应种类训练数据数量数
            num_c_feature[(int(trainlabel[i]),int(traindata[i][0]))] += 1
            #为离散特征feature0和不同种类统计训练数据数量
        
            for j in range(1,8):
                if subset_array_cnt[(int(trainlabel[i]),j)] == 0:
                    #建立数组sub_array
                    subset_array_dict[(int(trainlabel[i]),j)] = np.array(float(traindata[i][j]))
                    subset_array_cnt[(int(trainlabel[i]),j)] += 1
                else:
                    subset_array_dict[(int(trainlabel[i]),j)] = np.append(subset_array_dict[(int(trainlabel[i]),j)],float(traindata[i][j]))
        

        # debug
        # print(subset_array_dict)
        # 计算PC
        for i in range(1,4):
            self.Pc[i] = (num_c[i]+1)/(num_c[1]+num_c[2]+num_c[3]+3)
        # 计算Px[0]c
        for i in range(1,4):
            for j in range(0,8):
                if j == 0:
                    # 离散条件概率
                    for k in range(1,4):
                        self.Pxc[(i,j,k)] = (num_c_feature[i,k]+1)/(num_c[i]+3)
                else:
                    # 连续条件概率
                    self.Pxc[(i,j)] = self.mean_and_standard_deviation(subset_array_dict[(i,j)])


    def norm_distribution_function(self,mean,s_d,x):
        temp = ((2*math.pi)**0.5)*s_d
        # print(temp)
        temp = 1/temp
        exp = math.exp(-0.5*((x-mean)**2)/(s_d**2))
        temp = temp*exp
        return temp
    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''       
        pred = []
        test_num = features.shape[0]
        for k in range(test_num):
            max = 0
            c_predict = 0
            probabilities = []
            for c in range(1,4):
                temp = self.Pc[c]
                temp *= self.Pxc[(c,0,int(features[k][0]))]
                for i in range(1,8):
                    (mean,s_d) = self.Pxc[(c,i)]
                    p = self.norm_distribution_function(mean,s_d,features[k][i])
                    temp *= p
                '''
                probabilities.append(temp)
            c_predict = np.argmax(probabilities)
                '''
            
                if temp > max:
                    max = temp
                    c_predict = c
            
            pred.append(c_predict)
        pred = np.array(pred).reshape(test_num,1)
        return pred
                

def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    # debug
    print(Nayes.Pc)
    print(Nayes.Pxc)
    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果
    # print(pred)
    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()