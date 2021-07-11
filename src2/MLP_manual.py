import torch
import math
import numpy


class MLP:

    def __init__(self,numInputNodes,numHiddenNodes1,numHiddenNodes2,numOutputNodes,lr,epochs):
        # pass
        self.numInputNodes = numInputNodes
        self.numHiddenNodes1 = numHiddenNodes1
        self.numHiddenNodes2 = numHiddenNodes2
        self.numOutputNodes = numOutputNodes
        self.lr = lr
        self.epochs = epochs

        # key_point:初始权重的设置
        # 切记不能全部设置成0；正态分布设置要让方差小一点
        # 经验规则：在一个节点传入链接数量平方根倒数的范围内随机采样，即从均值为0、标准方差等于节点传入链接数量平方根倒数的正态分布中进行采样。
        self.weightInputHidden1 = numpy.random.normal(0.0, pow(self.numHiddenNodes1, -0.5),(self.numHiddenNodes1, self.numInputNodes))
        self.weightHidden1Hidden2 = numpy.random.normal(0.0, pow(self.numHiddenNodes2, -0.5),(self.numHiddenNodes2, self.numHiddenNodes1))
        self.weightHidden2Output = numpy.random.normal(0.0, pow(self.numOutputNodes, -0.5),(self.numOutputNodes, self.numHiddenNodes2))
        
        # 初始化所有bias均为0
        self.biasHidden1 = numpy.zeros((1,self.numHiddenNodes1)).T
        self.biasHidden2 = numpy.zeros((1,self.numHiddenNodes2)).T
        self.biasOutput = numpy.zeros((1,numOutputNodes)).T

    
    def activation_func(self,z):
        # pass
        return 1.0/(1.0+math.exp((-1)*z))
        
    

    def query(self,train_data):
        # pass
        # query函数表示通过当前参数的神经网络（MLP)，查询训练数据的输出结果
        train_num = train_data.shape[0]
        inputs = train_data.T


        # input -- hidden layer1
        hidden1_inputs = numpy.dot(self.weightInputHidden1, inputs)


        # 这个diag的用法存疑
        hidden1_outputs = numpy.diag(self.biasHidden1) * numpy.ones((self.numHiddenNodes1,train_num))
        hidden1_outputs = hidden1_inputs + hidden1_outputs

        # 通过激活函数处理
        for i in range(hidden1_outputs.shape[0]):
            for j in range(hidden1_outputs.shape[1]):
                hidden1_outputs[i,j] = self.activation_func(hidden1_outputs[i,j])
        
        # hidden layer1 -- hidden layer2
        hidden2_inputs = numpy.dot(self.weightHidden1Hidden2, hidden1_outputs)
        hidden2_outputs = numpy.diag(self.biasHidden2) * numpy.ones((self.numHiddenNodes2,train_num))
        hidden2_outputs = hidden2_outputs + hidden2_inputs

        for i in range(hidden2_outputs.shape[0]):
            for j in range(hidden2_outputs.shape[1]):
                hidden2_outputs[i,j] = self.activation_func(hidden2_outputs[i,j])

        # hidden layer2 -- output : softmax layer
        output_inputs = numpy.dot(self.weightHidden2Output, hidden2_outputs)
        output_outputs = numpy.diag(self.biasOutput) * numpy.ones((self.numOutputNodes,train_num))
        output_outputs = output_outputs + output_inputs

        for i in range(output_outputs.shape[0]):
            for j in range(output_outputs.shape[1]):
                output_outputs[i,j] = numpy.exp(output_outputs[i,j])
        
        #按列求和
        output_colsum = numpy.sum(output_outputs,axis = 0)

        for i in range(output_outputs.shape[0]):
            for j in range(output_outputs.shape[1]):
                output_outputs[i,j] = output_outputs[i,j] / output_colsum[j]


        # output_outputs 为 （numOutputNodes * train_num）的矩阵
        return output_outputs

    def loss_func(self,outputs,labels):
        # cross-entropy loss function
        train_num = labels.shape[0]
        temp = numpy.zeros((1,train_num))
        for i in range(train_num):
            temp[0,i] = math.log(outputs[labels[i],i])
        return (-1*numpy.sum(temp)/train_num)


    def label_trans(self,labels):
        # 将输入的label（100*1）转化为（3*100）的新形式，便于进行后续反向传播误差处理
        train_num = labels.shape[0]
        temp = numpy.zeros((self.numOutputNodes,train_num))
        for i in range(train_num):
            temp[labels[i],i] = 1
        return temp


    def training(self,train_data,labels):
        # pass
        # 即调用query函数和loss_func函数，再实现back_propogation算法
        cur_epoch = 0
        train_num = train_data.shape[0]
        inputs = train_data.T
        loss = []
        # 当前神经网络训练轮数
        while(cur_epoch < self.epochs):
            cur_epoch = cur_epoch + 1
            '''PART1：正向计算结果'''

            # input -- hidden layer1
            hidden1_inputs = numpy.dot(self.weightInputHidden1, inputs)


            # 这个diag的用法存疑
            hidden1_outputs = numpy.diag(self.biasHidden1) * numpy.ones((self.numHiddenNodes1,train_num))
            hidden1_outputs = hidden1_inputs + hidden1_outputs

            # 通过激活函数处理
            for i in range(hidden1_outputs.shape[0]):
                for j in range(hidden1_outputs.shape[1]):
                    hidden1_outputs[i,j] = self.activation_func(hidden1_outputs[i,j])
        
            # hidden layer1 -- hidden layer2
            hidden2_inputs = numpy.dot(self.weightHidden1Hidden2, hidden1_outputs)
            hidden2_outputs = numpy.diag(self.biasHidden2) * numpy.ones((self.numHiddenNodes2,train_num))
            hidden2_outputs = hidden2_outputs + hidden2_inputs

            for i in range(hidden2_outputs.shape[0]):
                for j in range(hidden2_outputs.shape[1]):
                    hidden2_outputs[i,j] = self.activation_func(hidden2_outputs[i,j])

            # hidden layer2 -- output : softmax layer
            output_inputs = numpy.dot(self.weightHidden2Output, hidden2_outputs)
            output_outputs = numpy.diag(self.biasOutput) * numpy.ones((self.numOutputNodes,train_num))
            output_outputs = output_outputs + output_inputs

            for i in range(output_outputs.shape[0]):
                for j in range(output_outputs.shape[1]):
                    output_outputs[i,j] = numpy.exp(output_outputs[i,j])

            # 按列求和
            output_colsum = numpy.sum(output_outputs,axis = 0)

            for i in range(output_outputs.shape[0]):
                for j in range(output_outputs.shape[1]):
                    output_outputs[i,j] = output_outputs[i,j] / output_colsum[j]
                    

            # 计算误差
            loss.append(self.loss_func(output_outputs,labels))

            '''PART2：反向传播误差'''
            temp = numpy.ones((train_num,1))
            
            # 计算Loss关于W3的梯度L_W3
            A3 = output_outputs
            A2 = hidden2_outputs
            Y = self.label_trans(labels)
            L_Z3 = A3 - Y
            L_W3 = numpy.dot(L_Z3,A2.T) 

            L_b3 = numpy.dot(L_Z3,temp)
            # 计算Loss关于W2的梯度L_W2
            # 先计算Loss关于Z2的梯度L_Z2
            A1 = hidden1_outputs
            W3 = self.weightHidden2Output
            L_Z2 = numpy.multiply(numpy.dot(W3.T,L_Z3),numpy.multiply(A2,numpy.ones((A2.shape[0],A2.shape[1]))-A2))
            L_W2 = numpy.dot(L_Z2,A1.T)

            L_b2 = numpy.dot(L_Z2,temp)
            # 注意这里multiply是hadamard product
            # 求导本质上相当于对其中每一项进行求导，也是一个hadamard product的运算

            # 计算Loss关于W1的梯度L_W1
            # 同样需要先计算Loss关于Z1的梯度L_Z1
            A0 = inputs
            W2 = self.weightHidden1Hidden2
            L_Z1 = numpy.multiply(numpy.dot(W2.T,L_Z2),numpy.multiply(A1,numpy.ones((A1.shape[0],A1.shape[1]))-A1))
            L_W1 = numpy.dot(L_Z1,A0.T)

            L_b1 = numpy.dot(L_Z1,temp)

            # 更新权值W[1..3] 和 b[1..3]（梯度下降）
            self.weightHidden2Output -= self.lr * L_W3
            self.weightHidden1Hidden2 -= self.lr * L_W2
            self.weightInputHidden1 -= self.lr * L_W1
            self.biasOutput -= self.lr * L_b3
            self.biasHidden2 -= self.lr * L_b2
            self.biasHidden1 -= self.lr * L_b1
        
        return loss









