import pandas as pd
import numpy as np
from collections import Counter

#读取csv数据，生成多维数组
def dataread(path):
    '''
    :param path: path is a string
    :return: array
    '''
    data_df = pd.read_csv(path)
    data_listT = np.array(data_df)
    return data_listT

#计算各特征的样本个数，以字典的形式输出
def label(data):
    '''
    :param data: array from dataread
    :return: 2 dictionaries include keys:labels and values:number of labels,such as{'high-no':2,… ｝,｛'no':5, ‘yes’:9}
    '''
    list = []
    for i in range(len(data.T[0])):
        for j in range(len(data[0])-1):
            list.append(str(data[i][j])+'-'+str(data[i][-1]))
    return dict(Counter(list)),dict(Counter(data.T[-1]))


#计算先验概率表，以同键字典输出
def priorpcalcu(dict1,dict2):
    '''
    :param dict1: Feature Samples and their num,{'high-no':2, …｝
    :param dict2: Events and their num,｛'no':5, ‘yes’:9}
    :return:2 dictionaries include Priori probability of each feature
    '''
    for key in list(dict1.keys()):
        if '-no' in key:
            dict1[key]=float(dict1[key]/dict2['no'])
        else:
            dict1[key]=float(dict1[key]/dict2['yes'])
    p = 0
    for key in list(dict2.keys()):
        p += dict2[key]
    for key in list(dict2.keys()):
        dict2[key] = dict2[key]/p
    return dict1,dict2

#将输入的x=<income,student,credit_rating>转化为列表，方便调用
def Sample_to_be_evaluated(income,student,credit_rating):
    sample1 = [income + '-' + 'yes', student + '-' + 'yes', credit_rating + '-' + 'yes', 'yes']
    sample2 = [income + '-' + 'no', student + '-' + 'no', credit_rating + '-' + 'no', 'no']
    return sample1,sample2

#计算条件影响下yes和no的概率和作决策
def computeprob(dict1,dict2,sample1,sample2):
    '''
    :param dict1: Priori probability1
    :param dict2: Priori probability2
    :param sample1:when x is yes
    :param sample2:when x is no
    :return:Probability_yes, Probability_no, judgement
    '''
    #预处理
    l1 = sample1
    l2 = sample2
    Probability_yes  = 1
    Probability_no = 1
    #求事件结果为yes的概率
    for i in range(len(l1)-1):
        key = l1[i]
        p = dict1[key]
        Probability_yes *= p
    Probability_yes *= dict2[l1[-1]]
    # 求事件结果为no的概率
    for j in range(len(l2)-1):
        key = l2[j]
        p = dict1[key]
        Probability_no *= p
    Probability_no *= dict2[l2[-1]]
    #根据yes和no概率大小作决策
    judgement = 'none'
    if Probability_yes>Probability_no:
        judgement = 'yes'
    else:
        judgement = 'no'
    return Probability_yes, Probability_no, judgement


#定义类、类方法
class Bayesian():
    def bayesianclassifier(self,datapath,income,student,credit_rating):
        self.datalist = dataread(datapath)
        self.label1,self.label2 = label(self.datalist)
        self.prior_p1,self.prior_p2 = priorpcalcu(self.label1,self.label2)
        self.list1,self.list2=Sample_to_be_evaluated(income, student, credit_rating)
        self.prob1,self.prob2,self.judgement = computeprob(self.prior_p1,self.prior_p2,self.list1,self.list2)
        return self.prob1,self.prob2,self.judgement

if __name__ == "__main__":
    bys = Bayesian() #类实例化
    P_yes,P_no,Judgement = bys.bayesianclassifier('C:/Users/ACER/Desktop/bayesian/BayesianData.csv','high','yes','excellent')#调用类方法，输入数据文件地址，条件x
    #打印结果
    print("Probability_yes is:",P_yes)
    print("Probability_no is:",P_no)
    print("judgement is:", Judgement)





