from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.000005,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs

    '''
    添加一个函数定义模块
    '''
    def loss(self,features,labels,w):
        attachment = np.ones(features.shape[0])
        X = np.c_[attachment,features]
        # X为原features首列改变成1的结果
        temp = np.dot(X,w)
        # print((temp-labels).shape)
        return int(np.dot((temp-labels).reshape(1,-1),temp-labels) + self.Lambda*np.dot(w.reshape(1,-1),w))
    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        ''''
        需要你实现的部分
        '''
        attachment = np.ones(train_features.shape[0])
        print(attachment.shape,train_features.shape)
        X = np.c_[attachment,train_features]
        w = np.zeros(train_features.shape[1] + 1)
        w = w.reshape(-1,1)
        # w的初值为0

        fit_epochs = self.epochs
        while fit_epochs > 0 :
            # print(w)
            # print(self.loss(train_features,train_labels,w))
            fit_epochs = fit_epochs - 1
            # temp = X*w-train_labels
            # print(X)
            temp = np.dot(X,w)
            temp = temp - train_labels
            # print(temp)
            temp = np.dot(temp.reshape(temp.shape[1],temp.shape[0]),X)
            # print(temp)
            grad = 2*temp + 2*self.Lambda*w.reshape(1,-1)
            # print(grad)
            # 计算梯度
            w = w - self.lr*grad.reshape(-1,1)
        # print(w)
        self.w = w
        
    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        ''''
        需要你实现的部分
        '''
        test_num = test_features.shape[0]
        attachment = np.ones(test_features.shape[0])
        X = np.c_[attachment,test_features]
        i = 0
        pred = []
        while i < test_num :
            y_pred = np.dot(X[i],self.w)
            # print(y_pred.shape,X[i].shape,self.w.shape)
            if y_pred < 1.5 :
                pred.append(1)
            elif y_pred > 2.5 :
                pred.append(3)
            else :
                pred.append(2)
            i = i+1
        pred = np.array(pred).reshape(test_num,1)
        return pred


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR=LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
