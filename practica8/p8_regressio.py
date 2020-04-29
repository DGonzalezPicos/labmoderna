import numpy as np
import scipy.constants as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from uncertainties import ufloat
from uncertainties.umath import *

# Data for the Electron Diffraction lab
# Lab performed on the 21st Feb. 2020 by I.Alsina and D.Gonzalez
#%%
# mesures dels diametres en cm
diam2 = np.array([2.46,2.28,2.16,2.02,1.92,1.78,1.75,1.72,1.64,1.58,1.43,1.27,1.32,1.23])
diam1 = np.array([4.58,4.35,3.96,3.67,3.47,3.32,3.19,3.04,2.93,2.87,2.74,2.62,2.42,2.31])
ddiam = 0.05 #incertesa en les mesures de diametres en cm

R = 6.5 #cm
theta1 = (1/4)*np.arcsin(diam1/(2*R))
theta2 = (1/4)*np.arcsin(diam2/(2*R))
dtheta1 = (1/4)*1/np.sqrt(1 - diam1/(2*R))*ddiam/(2*R)
dtheta2 = (1/4)*1/np.sqrt(1 - diam2/(2*R))*ddiam/(2*R)

E = np.array([3.1,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,9.0,10.,11.]) #keV
hc = 1.24e-6 # ev*m
x = (hc/np.sqrt(2*0.511*1e6*E*1e3))*1e12 #  pm
y1 = 2*np.sin(theta1)
y2 = 2*np.sin(theta2)
dy1 = 2*np.cos(theta1)*dtheta1
dy2 = 2*np.cos(theta2)*dtheta2

#%%

def linreg(x,y,dy):
    '''x,y data points (same dimensions),dy = uncertainty on y
    Performs a linear regression y = m*x + b
    Returns m and b with uncertainties and prints the R^2 value
     '''
    x = x.reshape((-1,1))
    N = len(x)

    #Linear Regression with scikit-learn

    model = LinearRegression(fit_intercept=True).fit(x, y) #if True, b is not fixed to 0
    y_reg = model.coef_*x + model.intercept_
    
    #Calculation of the uncertainties on (m,b), same results as with LINEST (excel)
    ss_yy = []
    ss_e = []
    ss_xx = []
    ss_xy = []
    
    for i in range(N):
        ss_yy.append((y[i]-y.mean())**2)
        ss_e.append((y[i]-y_reg[i])**2)
        ss_xx.append((x[i]-x.mean())**2)
        ss_xy.append((x[i]-x.mean())*(y[i]-y.mean()))
    ss_yy = sum(ss_yy)
    ss_e = sum(ss_e)
    ss_xx = sum(ss_xx)
    ss_xy = sum(ss_xy)
    
    s2_yx = ss_e/(N-2)
    
    s_m = np.sqrt(s2_yx/ss_xx) #slope stdev
    s_b = np.sqrt(s2_yx*((1/N)+(x.mean()**2)/ss_xx)) #b stdev
        
    #add uncertainties
    m_param = ufloat(model.coef_, s_m)
    b_param = ufloat(model.intercept_, s_b)  
    print('R^2 = %5.4f' % model.score(x,y))
    return m_param,b_param

def linplot(x,y,dy1,m,b,title):
    '''Arguments: x,y=data points, dy1=uncertainty on y, 
    (m,b)=fit_parameters, title= plot title (string)'
    Returns: Plot + saved figure (.png)
    '''
    f = m*x + b
    plt.plot(x ,f,label='Linear Fit')
    plt.plot(x,y,'bo', color='red',label= 'Data points')
    plt.title(title)
    plt.xlabel('$hc/\sqrt{2m_e c^2 E(eV)}$')
    plt.ylabel(r'$2 \sin{2\theta_1}$')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.errorbar(x,y,yerr=dy1, fmt='.k', label='$dy$')
#     plt.errorbar(x,y,xerr=0.0, label='$dx$', fmt='.k')
    plt.legend()
    title = title + '.png'
    plt.savefig( title,dpi=300)
    plt.show()


#%%

m1, b1 = linreg(x,y1,dy1)
print('Linear regression: y(x)=m*x + b')

print('m = ', m1)
print('b = ', b1)
fig1 = linplot(x,y1,dy1,m1.nominal_value,b1.nominal_value,'D1')

m2, b2 = linreg(x,y2,dy2)
print('Linear regression: y(x)=m*x + b')
print('m = ', m2)
print('b = ', b2)
fig2 = linplot(x,y2,dy2,m2.nominal_value,b2.nominal_value,'D2')
#%%
d1 = 1/m1 # m
d2 = 1/m2 # m
a = 2*d1/np.sqrt(3) # a is the atomic separation of atoms, aresta del hexagono
a_real = 142 #pm, hace falta referenciar el valor 

z = d2/d1
print('d1= ',(d1),'pm')
print('d2= ', (d2),'pm')
print('a = ',(a),'pm')
print('1/d1 = %3.3f pm-1' % (1/d1.nominal_value))
print('1/d2 = %3.3f pm-1' % (1/d2.nominal_value))
# Teoricamente, x deberia ser sqrt(3)
print('d2/d1 = ', z)
print('sqrt(3) = %3.3f' % np.sqrt(3))
print('Error relatiu en x en %:' ,(z-np.sqrt(3)/np.sqrt(3))*100)


#%% función para la celda siguiente
def plot_lambda(deBroglie, Bragg1, dy, m, b, title):
    plt.errorbar(deBroglie, Bragg1,yerr=dy, fmt='.k', ecolor='.7', label='Dades')
    plt.plot(deBroglie, m.n*x + b.n, label='y=mx+b' )
    plt.text(x=11.5,y=31, s='m = {:3.3f}'.format(m), fontsize = 15)
    
    plt.ylabel('$\lambda$ (Bragg)', fontsize = 16)
    plt.xlabel('$\lambda$ (DeBroglie)', fontsize = 16)
    plt.title(title)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc='upper left')
    plt.savefig(title+'comparacio_lambda',dpi=300)
    plt.show()
    return None
#%%
# Plotea la longitud de onda de De Broglie vs de Bragg

# E keV --> eV
deBroglie = 1e12*sp.h/np.sqrt(2*sp.m_e*sp.e*E*1e3) # en pm
Bragg1 = 2*d1.nominal_value*np.sin(theta1) # en pm
Bragg2 = 2*d2.nominal_value*np.sin(theta2) # en pm

# calculem l'incertesa en Bragg1 
dy1 = 2*np.sqrt((np.sin(theta1)**2*d1.n**2)+(d1.n*np.cos(theta1)*dtheta1)**2)
dy2 = 2*np.sqrt((np.sin(theta2)**2*d2.n**2)+(d2.n*np.cos(theta2)*dtheta2)**2)

m1, b1 = linreg(deBroglie, Bragg1, dy1)
m2, b2 = linreg(deBroglie, Bragg2, dy2)
# Plot
plot_lambda(deBroglie, Bragg1, dy1, m1, b1, 'd1'), plot_lambda(deBroglie, Bragg2, dy2, m2, b2, 'd2')

