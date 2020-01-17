import numpy as np
import numpy.linalg as lin
from sklearn.preprocessing import StandardScaler
import pandas as pd

class mypca(object):
    '''
    k : component 수
    n : 원래 차원
    components : 고유벡터 저장소 shape (k,n)
    explain_values : 고유값 shape (k,)
    '''
    n = None
    k = None
    components = None
    explain_values= None
    
    def __init__(self, k=None, X_train=None):
        '''
        k의 값이 initial에 없으면 None으로 유지
        '''
        if k is not None :
            self.k = k       
        if X_train is not None:
            self.fit(X_train)
            self.n = X_train.shape[1]
            
    def fit(self,X_train=None):
        if X_train is None:
            print('Input is nothing!')
            return
        if self.k is None:
            self.k = min(X_train.shape[0],X_train.shape[1])
            
        #############################################
        # TO DO                                     #
        # 인풋 데이터의 공분산행렬을 이용해         #
        # components와 explain_values 완성          # 
        #############################################
        
        #공분산 행렬
        train_matrix = pd.DataFrame(X_train).cov()

        #explain_values(고유값 저장)
        self.explain_values = lin.eig(train_matrix)[0]

        #components(고유벡터 저장)
        self.components = lin.eig(train_matrix)[1]
        
        #############################################
        # END CODE                                  #
        #############################################

        return (self.explain_values, self.components)
    
    def transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        
        result = None
        '''
        N : X의 행 수
        result의 shape : (N, k)
        '''
        #############################################
        # TO DO                                     #
        # components를 이용해 변환결과인            #
        # result 계산                               #
        #############################################

        components = self.components
        explain_values = list(self.explain_values)
        #고유값을 내림차순으로 정렬하여, 상위 k개만 뽑기
        k_sort_values = sorted(list(self.explain_values), reverse=True)[:int(self.k)]

        #고유값 상위 k개의 인덱스 찾기
        idx = []
        for i in k_sort_values:
            idx.append(explain_values.index(i))

        #고유값 상위 k개의 고유벡터 찾기
        components = []
        for i in idx:
            components.append(list(self.components[:, int(i)]))
        self.components = np.array(components)


        #고유값 상위 k개와 데이터를 내적해서 주성분 return하기
        components = self.components.transpose()
        result = np.dot(X, components)

        #############################################
        # END CODE                                  #
        #############################################       
        return result #150x2
    
    def fit_transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        self.fit(X)
        return self.transform(X)