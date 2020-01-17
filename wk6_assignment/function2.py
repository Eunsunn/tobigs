# forward batch normalization
def function1(x, gamma, beta, eps):

  N, D = x.shape #데이터개수, 피쳐 개수


  mu = 1./N * np.sum(x, axis = 0) #피쳐별 평균

 
  xmu = x - mu #편차

  
  sq = xmu ** 2 #편차제곱


  var = 1./N * np.sum(sq, axis = 0) #편차제곱합의 평균 = 분산 (피쳐별 분산)

  
  sqrtvar = np.sqrt(var + eps) #분산에 엡실론(아주작은 값)을 더하고 루트를 취함(나눌때 0으로 안나누기위해 보정한 표준편차)

  ivar = 1./sqrtvar # 1/보정표준편차

  
  xhat = xmu * ivar #(정규화한 x) = (x - 평균) / 표준편차

  
  gammax = gamma * xhat # (스케일링 파라미터)*정규화한 값  #차원은 X와 동일, 감마는 스케일링 파라미터

  out = gammax + beta  # (스케일링 파라미터)*정규화한 값 + (shift파라미터)   #(차원은 X와 동일)

  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
          #정규화한 x, scaling파라미터, (x-평균), 1/보정표준편차, 보정표준편차, 분산, 엡실론(아주작은 양수) : return
  return out, cache






# backpropagation batch normalization
def function2(dout, cache):

  
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache


  N,D = dout.shape 

  
  dbeta = np.sum(dout, axis=0) 
  dgammax = dout 

  dgamma = np.sum(dgammax*xhat, axis=0) 
  dxhat = dgammax * gamma


  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar


  dsqrtvar = -1. /(sqrtvar**2) * divar


  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar


  dsq = 1. /N * np.ones((N,D)) * dvar


  dxmu2 = 2 * xmu * dsq


  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)


  dx2 = 1. /N * np.ones((N,D)) * dmu


  dx = dx1 + dx2

  return dx, dgamma, dbeta  #dx:인풋데이터 x의 그레디언트, dgamma:스케일파라미터 감마 각각의 그레디언트, dbeta:shift파라미터 베타 각각의 그레디언트
