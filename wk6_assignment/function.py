
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient. #그레디언트의 지수가중평균 > 방향(momentum)
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient. #그레디언트제곱의 지수가중평균 > 속도(RMSprop)
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### layer의 각 W와 b마다 그레디언트 가중평균, 그레디언트 제곱의 가중평균 틀을 생성해놓음(gradient의 차원과 동일)
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    ### END CODE HERE ###
    
    return v, s


# Adam
def function1(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    L = len(parameters) // 2  #neural layer의 개수      
    v_corrected = {}                        
    s_corrected = {}                     
    
  
    for l in range(L):
        #layer의 개수만큼 momentum과 RMSProp 생성
        
       # v-그레디언트 지수가중평균, s-그레디언트 제곱 지수가중평균
       # adam의 Momentum(방향)
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db" + str(l+1)]
        

        # m_hat(편향보정)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-np.power(beta1,t))
       
        # adam의 RMSProp(속도)
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.square(grads["db" + str(l+1)])

        # s_hat(편향보정)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-np.power(beta2,t))

        # Momentum과 RMSProp로 모든 레이어의 W와 b 업데이트
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
 


    return parameters, v, s












# Dropout foward
def function2(X, parameters, keep_prob = 0.5):

    
    np.random.seed(1)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1                         # 첫번째 linear 값
    A1 = relu(Z1)                                   # linear 값을 relu에 넣은 결과
    D1 = np.random.rand(A1.shape[0],A1.shape[1])    # activation 차원과 같은 0과 1사이 값 생성               
    D1 = D1 < keep_prob                             # 0.5의 확률로 노드를 끄고 킴(결과는 0과 1)
    A1 = np.multiply(A1,D1)                         # A와 D를 곱해 activation 결과 노드를 0.5의 확률로 끄거나 킴(Dropout)
    A1 = A1/keep_prob                               # Dropout 결과 유지할 확률 p=keep_prob이므로 train set에서 기대값은 px가 된다.
                                                    # 하지만 test set에는 dropout을 하지않고, 그때의 기대값은 x가 되기때문에
                                                    # 기대값이 달라지는 것을 방지하기 위해 p로 나누어준다.
    
    Z2 = np.dot(W2, A1) + b2                        # 두번째 linear 값
    A2 = relu(Z2)                                   # linear 값을 relu에 넣은 결과
    
    D2 = np.random.rand(A2.shape[0],A2.shape[1])    # 두번째 activation 차원과 같은 0과 1사이 값 생성                    
    D2 = D2 < keep_prob                             # 0.5의 확률로 노드를 끄고 킴(결과는 0과 1)
    A2 = np.multiply(A2,D2)                         # activation 노드들을 0.5의 확률로 끄거나 킴(Dropout)
    A2 = A2/keep_prob                               
 
    Z3 = np.dot(W3, A2) + b3                        # 세번째 linear
    A3 = sigmoid(Z3)                                # sigmoid로 각 라벨로 분류할 확률을 받아내기
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache






# Dropout backpropagation
def function3(X, Y, cache, keep_prob):

    
    m = X.shape[1] #데이터의 개수
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y #Loss를 z3(마지막 linear값)으로 편미분(그레디언트)
    dW3 = 1./m * np.dot(dZ3, A2.T) #Loss를 W3(마지막 linear의 입력값)으로 편미분하고,
                                   #데이터 개수의 의존성을 낮추고, 데이터 개수라는 정보를 담기 위해 데이터개수로 나눠줌
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True) #Loss를 b3으로 편미분> gradient Z3(dZ3)에서 흐르기때문에 동일하나
                                                      #차원을 맞춰주기 위해 행으로 더함
    dA2 = np.dot(W3.T, dZ3) #gradient A2(gradient z3과) W3의 transpose를 곱함
    
    dA2 = np.multiply(dA2,D2) #0.5의 확률로 A2의 노드들을 임의로 끄거나 킨다.           
    dA2 = dA2/keep_prob       #유지할 확률로 나누어 보정한다.

    dZ2 = np.multiply(dA2, np.int64(A2 > 0)) #gradient Z2 = dA2 * ReLu미분
    dW2 = 1./m * np.dot(dZ2, A1.T) # gradient W2를 구하고 데이터 개수로 나눔
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True) #gradient b2(앞에서 흐르기때문에 그대로 써주고, 데이터 개수로 나눔)
    
    dA1 = np.dot(W2.T, dZ2) #gradient A1

    dA1 = np.multiply(dA1,D1) #0.5의 확률로 끄거나 킴  
    dA1 = dA1/keep_prob  #확률로 나누어 보정함

    dZ1 = np.multiply(dA1, np.int64(A1 > 0)) #gradient Z1(앞에서 흐르는 함수 * ReLu미분)
    dW1 = 1./m * np.dot(dZ1, X.T) #gradient W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True) #gradient b1
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients