import numpy as np
import scipy.constants as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from uncertainties import ufloat
from uncertainties.unumpy import *
#from uncertainties.umath import *

# Iteracions para el Nestle Sampling (con 1e3 es mas que suficiente)
npoints = int(1e4)  # si se quiere disminuir el tiempo poner n = 1e2 por ejemplo
#%% Datos del lab
# mesures dels diametres en cm
diam1 = np.array([2.46,2.28,2.16,2.02,1.92,1.78,1.75,1.72,1.64,1.58,1.43,1.27,1.32,1.23])
diam2 = np.array([4.58,4.35,3.96,3.67,3.47,3.32,3.19,3.04,2.93,2.87,2.74,2.62,2.42,2.31])
ddiam = 0.05 #incertesa en les mesures de diametres en cm

R = 6.5 #cm

E = np.array([3.1,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,9.0,10.,11.]) #keV

#%% Asignamos los errores a las magnitudes en arrays enteras
diam1err, diam2err = (np.array([]) for i in range(2))

for i in range(len(diam1)):
    diam1err = np.append(diam1err, ufloat(diam1[i],ddiam))
    diam2err = np.append(diam2err, ufloat(diam2[i],ddiam))

theta1 = (1/4)*arcsin(diam1err/R)
theta2 = (1/4)*arcsin(diam2err/R)

hc = 1.24e-6 # ev*m
x = (hc/np.sqrt(2*0.511*1e6*E*1e3))*1e12 #  pm
y1 = 2*sin(theta1)
y2 = 2*sin(theta2)
#%% split the nominal value and the error
y1nom, y2nom, y1s, y2s = (np.array([]) for i in range(4))
for i in range(len(y1)):
    y1nom, y2nom = np.append(y1nom, y1[i].n), np.append(y2nom, y2[i].n)
    y1s, y2s = np.append(y1s, y1[i].s), np.append(y2s, y2[i].s)


#%% Empleamos el package "nestle" para hacer las regresiones
import corner
import nestle

def nested_linear(x,y,yerr,b,title, npoints):
    """ Performs a linear regression with Nested Sampling and saves two plots
                                                                    1. data+fit
                                                                    2. corner plot
    Inputs: 
        x,y = np.arrays: of data and uncertainties in 
        yerr (np.array) = uncertainties on y
        b = np.array(b1,b2): boundaries of the parameters
        title = string: name for the plots' file .png
    Outputs:
        p: model parameter vector
        cov: covariance matrix (the diagonal elements are the errors of p)
    """
    
    def model(theta, x):
        m, c = theta
        return m*x + c 

    # The likelihood function:
    def loglike(theta):
        return -0.5*(np.sum((y-model(theta, x))**2/yerr**2))
    
    
    # Defines a flat prior in 0 < m < 1, 0 < c < 100:
    def prior_transform(theta):
        return np.array(b) * theta
    
    
    # Run nested sampling
    res = nestle.sample(loglike, prior_transform, 2, method='single',
                        npoints= npoints)
    print(res.summary())
    
    # weighted average and covariance:
    p, cov = nestle.mean_and_cov(res.samples, res.weights)
    
    print("m = {0:5.5f} +/- {1:5.5f}".format(p[0], np.sqrt(cov[0, 0])))
    print("b = {0:5.5f} +/- {1:5.5f}".format(p[1], np.sqrt(cov[1, 1])))
    
    plt.figure(figsize=(9,8))
    plt.errorbar(x, y, yerr=yerr, capsize=0, fmt='k.', ecolor='.7',label='Dades')
    plt.plot(x, model(p, x), c='k',label='Ajust lineal')
    plt.title(title,fontsize=14)
    plt.xlabel('$hc/\sqrt{2m_e c^2 E(eV)}$',fontsize=14)
    plt.ylabel(r'$2 \sin{2\theta_1}$',fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(title+'_nestle.png',dpi=300)
    plt.show()
    fig = corner.corner(res.samples, weights=res.weights, labels=['a', 'b'],
                        range=[0.99999, 0.99999],show_titles=True,
                        quantiles=(0.16, 0.84), 
                        levels=(1-np.exp(-0.5),),verbose=True,color='blue', 
                        bins=50, smooth=2,truths=p,
                        truth_color='green')
#    plt.savefig(title+'_corner.png',dpi=300)
    plt.show() 
    return p, cov
#%% Call the function to obtain the output
 
p1, cov1 = nested_linear(x,y1nom,y1s,np.array([10,-10]),'d1',  npoints)
p2, cov2 = nested_linear(x,y2nom, y2s, np.array([10,-10]),'d2',  npoints)
#%%
f = open('nestle_fit.txt', 'w')
f.write('# \t Nestle fit y = mx+b \n')
f.write('#    m +/- dm \t\t    b +/- db \n')
f.write('{0:5.5f}+/-{1:5.5f} \t {2:5.5f}+/-{3:5.5f} \n'.format(p1[0],np.sqrt(cov1[0][0]),p1[1],np.sqrt(cov1[1][1])))
f.write('{0:5.5f}+/-{1:5.5f} \t {2:5.5f}+/-{3:5.5f}'.format(p2[0],np.sqrt(cov2[0][0]),p2[1],np.sqrt(cov2[1][1])))

#%% Calcula las magnitudes del problema con los parametros del ajuste
# las pendientes de las rectas son 1/d1, 1/d2
d1 = 1/ufloat(p1[0],np.sqrt(cov1[0,0])) # UNIDADES: pm = 1e-12 m
d2 = 1/ufloat(p2[0],np.sqrt(cov2[0,0]))

a = 2*d1/np.sqrt(3) # a is the atomic separation of atoms, aresta del hexagono
a_real = 142 #pm, hace falta referenciar el valor 
print('a/a_real= {:2.2f}'.format(a/a_real)) # debería ser 1...


#%%
def nestle_linear2(x, y, yerr, b, title, npoints):
    def model(theta, x):
        m, c = theta
        return m*x + c 
    
    # The likelihood function:
    def loglike(theta):
        return -0.5*(np.sum((y-model(theta, x))**2/yerr**2))
    
    
    # Defines a flat prior in 0 < m < 1, 0 < c < 100:
    def prior_transform(theta):
        return np.array(b) * theta
    
    
    # Run nested sampling
    res = nestle.sample(loglike, prior_transform, 2, method='single',
                        npoints= npoints)
    print(res.summary())
    
    # weighted average and covariance:
    p, cov = nestle.mean_and_cov(res.samples, res.weights)

    print("m = {0:5.5f} +/- {1:5.5f}".format(p[0], np.sqrt(cov[0, 0])))
    print("b = {0:5.5f} +/- {1:5.5f}".format(p[1], np.sqrt(cov[1, 1])))
    # PLOT
    plt.errorbar(x, y, yerr=yerr, capsize=0, fmt='k.', ecolor='.7',label='Dades')
    plt.plot(x, model(p, x), c='k',label='y=mx+b')
    m = ufloat(p[0], np.sqrt(cov[0, 0]))
    plt.text(x=11.5,y=31, s='m = {:3.3f}'.format(m), fontsize = 15)
    
    plt.ylabel('$\lambda$ (Bragg)', fontsize = 16)
    plt.xlabel('$\lambda$ (DeBroglie)', fontsize = 16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc='upper left',fontsize=14)
    plt.title(title)
    plt.savefig(title + 'lambda_nestle.png',dpi=300)
    plt.show()
    fig = corner.corner(res.samples, weights=res.weights, labels=['a', 'b'],
                        range=[0.99999, 0.99999],show_titles=True,
                        quantiles=(0.16, 0.84), 
                        levels=(1-np.exp(-0.5),),verbose=True,color='blue', 
                        bins=50, smooth=2,truths=p,
                        truth_color='green')
#    plt.savefig(title + 'lambda_corner.png',dpi=300)
    plt.show() 
    return p, cov
#%%
# Plotea la longitud de onda de De Broglie vs de Bragg

# E keV --> eV
deBroglie = 1e12*sp.h/np.sqrt(2*sp.m_e*sp.e*E*1e3) # en pm
Bragg1 = 2*d1*sin(theta1) # en pm
Bragg2 = 2*d1*sin(theta1) # en pm


#split the nominal value and the error
braggnom1, braggs1, braggnom2, braggs2 = (np.array([]) for i in range(4))
for i in range(len(Bragg1)):
    braggnom1, braggs1 = np.append(braggnom1, Bragg1[i].n), np.append(braggs1, Bragg1[i].n)
    braggnom2, braggs2 = np.append(braggnom2, Bragg2[i].n), np.append(braggs2, Bragg2[i].n)
    
p1, cov1 = nestle_linear2(deBroglie, braggnom1, braggs1, np.array([10,-10]), 'd1',  npoints )
p2, cov2 = nestle_linear2(deBroglie, braggnom2, braggs2, np.array([10,-10]), 'd2',  npoints )