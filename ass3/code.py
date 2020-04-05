import numpy as np
import matplotlib.pyplot as plt



def load_data():
    """
    Return data in np format
    """

    from scipy.io import loadmat
    f1='drx5day_zone.csv.mat'
    f2='dwsdi_zone.csv.mat'
    return(loadmat(f1)['tn'],loadmat(f2)['tn'])



def lr(X,Y):
    """
    Returns slope and intercept of regression line
    """

    from sklearn import linear_model

    X=np.array(X).reshape((-1,1))
    Y=np.array(Y)
    regr=linear_model.LinearRegression()
    regr.fit(X,Y)
    m=regr.coef_[0]
    c=regr.intercept_
    return m,c



def part1():
    """
    Part-1: Linear Trend Analysis
    """

    global P,T

    # For Precipitation
    Ptrend=[]
    Ptime=[]
    for i in range(64):
        Ptime.append(1901+i)
        Ptrend.append(np.mean(P[:,:,i]))

    m,c=lr(Ptime,Ptrend)
    Y=[m*x+c for x in Ptime]
    plt.plot(Ptime,Ptrend)
    plt.plot(Ptime,Y)
    plt.xlabel('Year')
    plt.ylabel('Annual average precipitation')
    plt.show()

    # For Temperature
    Ttrend=[]
    Ttime=[]
    for i in range(64):
        Ttime.append(1951+i)
        Ttrend.append(np.mean(T[:,:,i]))

    m,c=lr(Ttime,Ttrend)
    Y=[m*x+c for x in Ttime]
    plt.plot(Ttime,Ttrend)
    plt.plot(Ttime,Y)
    plt.xlabel('Year')
    plt.ylabel('Annual average temperature')
    plt.show()



def MK(X,Y):
    """
    Returns MK value for array Y
    """

    ret=0
    t1=Y[:,None]>Y
    t2=np.multiply(X,t1)
    ret+=np.sum(t2)
    t1=Y[:,None]<Y
    t2=np.multiply(X,t1)
    ret-=np.sum(t2)
    return np.sign(ret)



def plot_hm(X,T):
    """
    Plot heat map
    """

    import seaborn as sns

    X=X.transpose()
    X=np.flipud(X)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False)
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,vmin=1951,vmax=2014)
    plt.title(T)
    plt.show()



def part2():
    """
    Part-2: Monotonic Trend Analysis using mann-kandall test
    """

    global P,T

    # For Precipitation
    PMK=np.zeros(121*121).reshape((121,121))
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i in range(121):
        for j in range(121):
            PMK[i][j]=MK(x,P[i][j])
    plot_hm(PMK,"Mann-kandall Test Analysis for Precipitation")

    # For Temperature
    TMK=np.zeros(121*121).reshape((121,121))
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i in range(121):
        for j in range(121):
            TMK[i][j]=MK(x,T[i][j])
    plot_hm(TMK,"Mann-kandall Test Analysis for Temperature")



def SS(X,Y,Z):
    """
    Returns SS value for array Y
    """
    
    from sklearn.metrics import pairwise_distances
    
    ret=0
    Y2=Y.reshape((-1,1))
    np.place(Z,Z==0,1)

    t1=Y[:,None]>Y
    t2=np.multiply(X,t1)
    t3=pairwise_distances(Y2)
    t4=np.multiply(t2,t3)
    t5=np.divide(t4,Z)
    ret+=np.sum(t5)

    t1=Y[:,None]<Y
    t2=np.multiply(X,t1)
    t3=pairwise_distances(Y2)
    t4=np.multiply(t2,t3)
    t5=np.divide(t4,Z)
    ret-=np.sum(t5)

    # print(ret)
    return ret



def part3():
    """
    Part-3: Sen's slope test
    """

    global P,T

    # For Precipitation
    PSS=np.zeros(121*121).reshape((121,121))
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    y=np.array([i-j for i in range(64) for j in range(64)]).reshape((64,64))    
    for i in range(121):
        for j in range(121):
            PSS[i][j]=SS(x,P[i][j],y)
    plot_hm(PSS,"Sen's slope test for Precipitation")

    # For Temperature
    TSS=np.zeros(121*121).reshape((121,121))
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    y=np.array([i-j for i in range(64) for j in range(64)]).reshape((64,64))    
    for i in range(121):
        for j in range(121):
            TSS[i][j]=SS(x,T[i][j],y)
    plot_hm(TSS,"Sen's slope test for Temperature")



def PT(X,Y,b1):
    """
    Return change point for array X
    """

    n=X.shape[0]
    CP=[]
    for i in range(n):
        # for j in range(i):
            # for k in range(j+1,n):
                # temp+=np.sign(X[j]-X[k])
        b2=np.array([j%64<i for j in range(64*64)]).reshape((64,64))
        b3=np.multiply(b1,b2)
        ret=0
        t1=X[:,None]>X
        t2=np.multiply(b3,t1)
        ret+=np.sum(t2)
        t1=X[:,None]<X
        t2=np.multiply(b3,t1)
        ret-=np.sum(t2)
        CP.append(ret)
    
    T=np.argmax(CP)
    K=CP[T]
    sl=0.5
    t=64
    p=2*np.exp(-6*K**2/(t**3+t**2))

    # print(p)

    if(p<sl):
        return T+Y
    else:
        return 0



def part4():
    """
    Part-4: Change point detection test
    """

    global P,T

    # # For Precipitation
    # x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    # PCP=np.zeros(121*121).reshape((121,121))
    # for i in range(121):
    #     for j in range(121):
    #         print(i,j)
    #         PCP[i][j]=PT(P[i][j],1901,x)
    # plot_hm(PCP,"Change Point Test Analysis for Precipitation")

    # For Temperature
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    TCP=np.zeros(121*121).reshape((121,121))
    for i in range(121):
        for j in range(121):
            print(i,j)
            TCP[i][j]=PT(T[i][j],1951,x)
    plot_hm(TCP,"Change Point Test Analysis for Temperature")



if __name__ == "__main__":
    
    P,T=load_data()
    P=P[:,:,0:64]
    # print(P.shape,T.shape)
    # (121, 121, 64) (121, 121, 64)

    # part1()
    # part2()
    # part3()
    part4()
