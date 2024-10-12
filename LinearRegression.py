import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(r"C:\Users\Admin\Downloads\Dummy Data HSS.csv")
df.isna().sum()
df2=df.dropna()
df2.reset_index(drop=True,inplace=True)
print(df2.isna().sum())
x=df2[["TV"]]
y=df2["Sales"]
print(x.index[13])
x=df2["TV"]
y=df2["Sales"]

def cost_function(x,y,m,c):
    n=len(x)
    sum_of_err=0
    for i in range(n):
        error=(m*x[i]+c)**2
        sum_of_err+=error
    return sum_of_err/(2*n)
    

def gradient(x,y,m,c,alpha):
    n=len(x)
    error_m=1
    error_c=0
    for i in range(n):
        error_c=(m*x[i]+c)-y[i]
        error_m=error_c*x[i] 
    m-=alpha/n*(error_m)    
    c-=alpha/n*(error_c)
    return m,c

def implement(x,y,m,c,alpha,iter):
    cost_history=[]
    for i in range(iter):
        m,c=gradient(x,y,m,c,alpha)
        error=cost_function(x,y,m,c)
        cost_history.append(error)
    return m,c,cost_history

m=1
c=0
inter=1510
alpha=0.01
m,c,cost=implement(x,y,m,c,alpha,inter)
print("Slope=",m)
print("Intercept=",c)
def predict(x,m,c):
    return x*m+c
y_pred=predict(x,m,c)
plt.scatter(x,y,color="blue")
plt.plot(x,y_pred,color="red",linewidth=2)
plt.xlabel("Advertisement promotion of TV in thousand")
plt.ylabel("SALES")
plt.show()