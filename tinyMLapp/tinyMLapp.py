import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
    return sp.sum((f(x)-y)**2)

data=sp.genfromtxt("web_traffic.tsv",delimiter='\t')
print(data[:10])
print (data.shape)

x=data[:,0]
y=data[:,1]

print(sp.sum(sp.isnan(y)))   #check how many nan value present
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]
plt.scatter(x,y,linewidths=1)
plt.title("Web traffic over the month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
           ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()

              # oder1 model
fp1,residual,rank,sv,rcond=sp.polyfit(x,y,1,full="true")
print("model parameters are %s" %fp1)
print("residuals %s" %residual)
f1=sp.poly1d(fp1)    #modeling our paramets fp1 to an order 1 function
print(error(f1,x,y))

plt.scatter(x,y,linewidths=1)
plt.title("Web traffic over the month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
           ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
fx=sp.linspace(0,x[-1],1000)
plt.plot(fx,f1(fx),color='r',linewidth=1)
plt.legend(["d=%i" % f1.order], loc="upper left")



#order 2 model
fp2,residual,rank,sv,rcond=sp.polyfit(x,y,2,full="true")
f2=sp.poly1d(fp2)
plt.legend(["d=%i" % f2.order], loc="upper left")

plt.plot(fx,f2(fx),color='g',linewidth=2)
print(error(f2,x,y))
print("difference between order 1 and 2 =" ,error(f1,x,y)-error(f2,x,y))
plt.grid()
plt.show()

#further increasing *may order causes overfitting(error reduces but model can be used to predict new values)