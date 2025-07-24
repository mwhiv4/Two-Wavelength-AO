import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv,gamma
from scipy.integrate import quad
from mpmath import meijerg

#####################################################################
def GAMMA(As,Bs,dim): 
    return np.prod(gamma(As),axis=dim)/np.prod(gamma(Bs),axis=dim)

def MG_HelperA(D,z,k1,k2,i):
    a2 = z/k1
    a3 = z/k2
    As = [[23/12,3/4,7/12,1/4,1/12],[]]
    Bs = [[i/2+1/2,i/2+1,23/12],[-i/2,-i/2-1/2,1/2,0]]
    MG2 = meijerg(As,Bs,(D**2/(8*a2))**2,maxterms=10**5)
    MG3 = meijerg(As,Bs,(D**2/(8*a3))**2,maxterms=10**5)
    return 1/2*a2**(11/6)*MG2+1/2*a3**(11/6)*MG3

def MG_HelperB1(i,m,j,arg):
    As = [[-2*m-j+17/6,-m+3/2,-m+1,1/2],[-2*m+j+17/6]]
    Bs = [[i+1,-m+7/3,-m+17/6],[0,-i-1]]
    MG = meijerg(As,Bs,arg,maxterms=10**5)
    return MG

def MG_HelperB2(i,m,j,arg):
    As = [[-m/2-j/2+23/12,3/4,1/4,m/2+j/2+7/12,m/2+j/2+1/12],[-m/2-j/2+17/12]]
    Bs = [[i/2+1/2,i/2+1,m/2+j/2+17/12,m/2+j/2+23/12],[-i/2,-i/2-1/2,1/2,0]]
    MG = meijerg(As,Bs,arg,maxterms=10**5)
    return MG

def zeta_integrand_PR(zeta,Cn2,z,D,rhoD,k1,k2,low_lim,up_lim):
    subints = 50
    if np.isinf(up_lim) != 1:
        int_lims = np.linspace(low_lim,up_lim,int((up_lim-low_lim)/20))
        if len(int_lims) > 50:
            subints = len(int_lims)
        f,err = quad(kappa_integrand_PR,low_lim,up_lim,args=(zeta,Cn2,z,D,rhoD,k1,k2),limit=subints,points=int_lims)
    else:
        f,err = quad(kappa_integrand_PR,low_lim,up_lim,args=(zeta,Cn2,z,D,rhoD,k1,k2),limit=subints)
    return f

def kappa_integrand_PR(kappa,zeta,Cn2,z,D,rhoD,k1,k2):
    a1 = 0
    a2 = z/k1
    a3 = z/k2
    a4 = z/2*(1/k1-1/k2)
    a5 = z/2*(1/k1+1/k2)
    f1 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a1*zeta/z*(1-zeta/z)*kappa**2) \
        *(jv(1,D/2*zeta/z*kappa)**2-1/4*(D/2*zeta/z*kappa)**2)
    f2 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a2*zeta/z*(1-zeta/z)*kappa**2) \
        *(jv(1,D/2*zeta/z*kappa)**2-1/4*(D/2*zeta/z*kappa)**2)
    f3 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a3*zeta/z*(1-zeta/z)*kappa**2) \
        *(jv(1,D/2*zeta/z*kappa)**2-1/4*(D/2*zeta/z*kappa)**2)
    f4 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a4*zeta/z*(1-zeta/z)*kappa**2) \
        *(jv(1,D/2*zeta/z*kappa)**2-1/4*(D/2*zeta/z*kappa)**2)*jv(0,(1-zeta/z)*rhoD*kappa)
    f5 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a5*zeta/z*(1-zeta/z)*kappa**2) \
        *(jv(1,D/2*zeta/z*kappa)**2-1/4*(D/2*zeta/z*kappa)**2)*jv(0,(1-zeta/z)*rhoD*kappa)
    f = -2**(14/3)*5/9*np.sqrt(np.pi)*GAMMA([5/6],[2/3],0)*Cn2*D**(-2) \
        *(f1+1/2*f2+1/2*f3-f4-f5)
    return f        

def zeta_integrand_Z(zeta,Cn2,z,D,rhoD,phiD,i,j,opt,k1,k2,low_lim,up_lim):
    subints = 50
    if np.isinf(up_lim) != 1:
        int_lims = np.linspace(low_lim,up_lim,int((up_lim-low_lim)/20))
        if len(int_lims) > 50:
            subints = len(int_lims)
        f,err = quad(kappa_integrand_Z,low_lim,up_lim,args=(zeta,Cn2,z,D,rhoD,phiD,i,j,opt,k1,k2),limit=subints,points=int_lims)
    else:
        f,err = quad(kappa_integrand_Z,low_lim,up_lim,args=(zeta,Cn2,z,D,rhoD,phiD,i,j,opt,k1,k2),limit=subints)
    return f

def kappa_integrand_Z(kappa,zeta,Cn2,z,D,rhoD,phiD,i,j,opt,k1,k2):
    a1 = 0
    a2 = z/k1
    a3 = z/k2
    a4 = z/2*(1/k1-1/k2)
    a5 = z/2*(1/k1+1/k2)
    f1 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a1*zeta/z*(1-zeta/z)*kappa**2) \
        *jv(i+1,D/2*zeta/z*kappa)**2
    f2 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a2*zeta/z*(1-zeta/z)*kappa**2) \
        *jv(i+1,D/2*zeta/z*kappa)**2
    f3 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a3*zeta/z*(1-zeta/z)*kappa**2) \
        *jv(i+1,D/2*zeta/z*kappa)**2
    if j == 0:
        f4 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a4*zeta/z*(1-zeta/z)*kappa**2) \
            *jv(i+1,D/2*zeta/z*kappa)**2*jv(0,(1-zeta/z)*rhoD*kappa)
        f5 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a5*zeta/z*(1-zeta/z)*kappa**2) \
            *jv(i+1,D/2*zeta/z*kappa)**2*jv(0,(1-zeta/z)*rhoD*kappa)        
    else:
        if opt == 'x':
            f4 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a4*zeta/z*(1-zeta/z)*kappa**2) \
                *jv(i+1,D/2*zeta/z*kappa)**2*(jv(0,(1-zeta/z)*rhoD*kappa) 
                +(-1)**j*jv(2*j,(1-zeta/z)*rhoD*kappa)*np.cos(2*j*phiD))
            f5 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a5*zeta/z*(1-zeta/z)*kappa**2) \
                *jv(i+1,D/2*zeta/z*kappa)**2*(jv(0,(1-zeta/z)*rhoD*kappa) 
                +(-1)**j*jv(2*j,(1-zeta/z)*rhoD*kappa)*np.cos(2*j*phiD))
        else:
            f4 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a4*zeta/z*(1-zeta/z)*kappa**2) \
               *jv(i+1,D/2*zeta/z*kappa)**2*(jv(0,(1-zeta/z)*rhoD*kappa) 
               -(-1)**j*jv(2*j,(1-zeta/z)*rhoD*kappa)*np.cos(2*j*phiD))
            f5 = (zeta/z)**(-2)*kappa**(-14/3)*np.cos(a5*zeta/z*(1-zeta/z)*kappa**2) \
               *jv(i+1,D/2*zeta/z*kappa)**2*(jv(0,(1-zeta/z)*rhoD*kappa) 
               -(-1)**j*jv(2*j,(1-zeta/z)*rhoD*kappa)*np.cos(2*j*phiD))            
    f = 2**(14/3)*5/9*np.sqrt(np.pi)*GAMMA([5/6],[2/3],0)*(i+1)*Cn2*D**(-2) \
        *(f1+1/2*f2+1/2*f3-f4-f5)
    return f        
#####################################################################

Cn2 = 5e-15
D = 0.5
z = 5e3
i = 1
j = 1

wvlT = 2e-6
wvlB = np.linspace(500e-9,5e-6,6)     
kT = 2*np.pi/wvlT
kB = 2*np.pi/wvlB
theta0 = (-np.sqrt(np.pi)/8*GAMMA([-5/6],[2/3],0)*Cn2*kT*kB*z**(8/3))**(-3/5)
rhoD = z*np.mean(theta0)*np.logspace(-3,3,100)  
rhoD_quad = np.logspace(np.log10(np.min(rhoD)),np.log10(np.max(rhoD)),15)     
phiD = -np.pi+2*np.pi*np.random.uniform()
    
######################## Theory #####################################
PR_GI = np.zeros((len(wvlB),len(rhoD)))
PR_GIV = np.zeros((len(wvlB),len(rhoD)))
PTR_GI = np.zeros((len(wvlB),len(rhoD)))
PTR_GIV = np.zeros((len(wvlB),len(rhoD)))
sw_quad_PR = np.zeros((len(wvlB),len(rhoD_quad)))
sw_quad_PTR = np.zeros((len(wvlB),len(rhoD_quad)))
for ww in np.arange(len(wvlB)):
    kB = 2*np.pi/wvlB[ww]
    MGA = np.frompyfunc(MG_HelperA,5,1)
    A_PR = -2**(-7/2)/np.sqrt(np.pi)*5/9*GAMMA([5/6,-5/12,1/12,7/6,5/3,4/3,11/6],[23/12,29/12,17/12,23/12,2/3,11/3],0)*Cn2*z*D**(5/3) \
        -2**(2/3)*np.sqrt(np.pi)/3*GAMMA([5/6,7/12,11/6,17/12],[2/3,11/3],0)*Cn2*z*1/2*((z/kB)**(5/6)+(z/kT)**(5/6)) \
        -4/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*Cn2*z*D**(-2) \
        *np.array(MGA(D,z,kB,kT,0),dtype='float')
    A_Zi = 3/22*GAMMA([i-5/6,7/3],[2/3,i+23/6],0)*(i+1)*Cn2*z*D**(5/3) \
        +4/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2) \
        *np.array(MGA(D,z,kB,kT,i),dtype='float')
    # ===== rhoD > 2 alpha_k ===== #
    # - Group I two-poles
    MGB1 = np.frompyfunc(MG_HelperB1,4,1)
    a4 = z/2*np.abs(1/kB-1/kT)
    a5 = z/2*np.abs(1/kB+1/kT)
    B1 = 0
    B2 = 0
    for m in np.arange(2):
        B1 = B1-2**(-1/3)/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*Cn2*z*rhoD**(5/3) \
            *(-16/rhoD**4)**m/gamma(2*m+1)*(a4**(2*m)+a5**(2*m)) \
            *GAMMA([-m+4/3,-m+11/6,2*m-5/6,m+1/2,m+1],[-2*m+11/6],0)
        B2 = B2+2**(5/3)/np.pi*5/9*GAMMA([5/6],[2/3,11/3],0)*Cn2*z*D**(-2)*rhoD**(11/3) \
            *(-16/rhoD**4)**m/gamma(2*m+1)*(a4**(2*m)+a5**(2*m)) \
            *np.array(MGB1(0,m,0,(D/rhoD)**2),dtype='complex')
    PR_GI[ww,:] = A_PR+np.real(B1+B2)
    B = 0
    Bx = 0
    By = 0
    for m in np.arange(3):            
        if j == 0:
            B = B-2**(5/3)/np.pi*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2)*rhoD**(11/3) \
                *(-16/rhoD**4)**m/gamma(2*m+1)*(a4**(2*m)+a5**(2*m)) \
                *np.array(MGB1(i,m,0,(D/rhoD)**2),dtype='complex')
            Zi_GI = A_Zi+np.real(B)
        else:
            Bx = Bx-2**(5/3)/np.pi*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2)*rhoD**(11/3) \
                *(-16/rhoD**4)**m/gamma(2*m+1)*(a4**(2*m)+a5**(2*m)) \
                *(np.array(MGB1(i,m,0,(D/rhoD)**2),dtype='complex') \
                +(-1)**j*np.cos(2*j*phiD)*np.array(MGB1(i,m,j,(D/rhoD)**2),dtype='complex'))
            By = By-2**(5/3)/np.pi*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2)*rhoD**(11/3) \
                *(-16/rhoD**4)**m/gamma(2*m+1)*(a4**(2*m)+a5**(2*m)) \
                *(np.array(MGB1(i,m,0,(D/rhoD)**2),dtype='complex') \
                -(-1)**j*np.cos(2*j*phiD)*np.array(MGB1(i,m,j,(D/rhoD)**2),dtype='complex'))
            xZi_GI = A_Zi+np.real(Bx)
            yZi_GI = A_Zi+np.real(By)    
    PTR_GI[ww,:] = PR_GI[ww,:]-(xZi_GI+yZi_GI)
    # ============================ #
    # ===== rhoD < 2 alpha_k ===== #
    # - Groups IV & V two-poles
    MGB2 = np.frompyfunc(MG_HelperB2,4,1)
    a4 = z/2*np.abs(1/kB-1/kT)
    a5 = z/2*np.abs(1/kB+1/kT)
    B1 = 0
    B2 = 0
    for m in np.arange(3):
        B1 = B1-2/np.pi*5/9*Cn2*z*GAMMA([5/6,3/4,5/4],[2/3,11/3],0) \
            *(a4**(5/6)*(-rhoD**2/(2*a4))**m/gamma(m+1)**2 \
            *GAMMA([m/2+11/12,m/2+17/12,m/2-5/12,-m/2+11/12,-m/2+17/12], \
            [-m/2+11/12],0) \
            +a5**(5/6)*(-rhoD**2/(2*a5))**m/gamma(m+1)**2 \
            *GAMMA([m/2+11/12,m/2+17/12,m/2-5/12,-m/2+11/12,-m/2+17/12], \
            [-m/2+11/12],0))
        B2 = B2+4/np.sqrt(np.pi)*5/9*Cn2*z*D**(-2)*GAMMA([5/6],[2/3,11/3],0) \
            *(a4**(11/6)*(-rhoD**2/(2*a4))**m/gamma(m+1)**2*np.array(MGB2(0,m,0,(D**2/(8*a4))**2),dtype='complex') \
            +a5**(11/6)*(-rhoD**2/(2*a5))**m/gamma(m+1)**2*np.array(MGB2(0,m,0,(D**2/(8*a5))**2),dtype='complex'))
    PR_GIV[ww,:] = A_PR+np.real(B1+B2)
    B = 0
    Bx = 0
    By = 0
    for m in np.arange(2):            
        if j == 0:
            B = B-4/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2) \
                *(a4**(11/6)*(-rhoD**2/(2*a4))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a4))**2),dtype='complex') \
                +a5**(11/6)*(-rhoD**2/(2*a5))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a5))**2),dtype='complex'))
            Zi_GIV = A_Zi+np.real(B)
        else:
            Bx = Bx-4/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2) \
                *(a4**(11/6)*(-rhoD**2/(2*a4))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a4))**2),dtype='complex') \
                +a4**(11/6)*(-rhoD**2/(2*a4))**(m+j)/(gamma(m+1)*gamma(m+2*j+1))*np.cos(2*j*phiD)*np.array(MGB2(i,m,j,(D**2/(8*a4))**2),dtype='complex')  
                +a5**(11/6)*(-rhoD**2/(2*a5))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a5))**2),dtype='complex') \
                +a5**(11/6)*(-rhoD**2/(2*a5))**(m+j)/(gamma(m+1)*gamma(m+2*j+1))*np.cos(2*j*phiD)*np.array(MGB2(i,m,j,(D**2/(8*a5))**2),dtype='complex'))
            By = By-4/np.sqrt(np.pi)*5/9*GAMMA([5/6],[2/3,11/3],0)*(i+1)*Cn2*z*D**(-2) \
                *(a4**(11/6)*(-rhoD**2/(2*a4))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a4))**2),dtype='complex') \
                -a4**(11/6)*(-rhoD**2/(2*a4))**(m+j)/(gamma(m+1)*gamma(m+2*j+1))*np.cos(2*j*phiD)*np.array(MGB2(i,m,j,(D**2/(8*a4))**2),dtype='complex')  
                +a5**(11/6)*(-rhoD**2/(2*a5))**m/gamma(m+1)**2*np.array(MGB2(i,m,0,(D**2/(8*a5))**2),dtype='complex') \
                -a5**(11/6)*(-rhoD**2/(2*a5))**(m+j)/(gamma(m+1)*gamma(m+2*j+1))*np.cos(2*j*phiD)*np.array(MGB2(i,m,j,(D**2/(8*a5))**2),dtype='complex'))
            xZi_GIV = A_Zi+np.real(Bx)
            yZi_GIV = A_Zi+np.real(By)
    PTR_GIV[ww,:] = PR_GIV[ww,:]-(xZi_GIV+yZi_GIV)
    # ============================ #
    # ===== Quadrature ===== #
    kappa_lims = [0,1000]    
    sw_quad_Zi = np.zeros(np.shape(rhoD_quad))
    sw_quad_xZi = np.zeros(np.shape(rhoD_quad))
    sw_quad_yZi = np.zeros(np.shape(rhoD_quad))
    for rr in np.arange(len(rhoD_quad)):
        sw_quad_PR[ww,rr],err = quad(zeta_integrand_PR,0,z, \
            args=(Cn2,z,D,rhoD_quad[rr],kB,kT,kappa_lims[0],kappa_lims[1]))
        if j == 0:
            sw_quad_Zi[rr],err = quad(zeta_integrand_Z,0,z, \
                args=(Cn2,z,D,rhoD_quad[rr],phiD,i,j,'x',kB,kT,kappa_lims[0],kappa_lims[1]))
        else:
            sw_quad_xZi[rr],err = quad(zeta_integrand_Z,0,z, \
                args=(Cn2,z,D,rhoD_quad[rr],phiD,i,j,'x',kB,kT,kappa_lims[0],kappa_lims[1]))
            sw_quad_yZi[rr],err = quad(zeta_integrand_Z,0,z, \
                args=(Cn2,z,D,rhoD_quad[rr],phiD,i,j,'y',kB,kT,kappa_lims[0],kappa_lims[1]))
    sw_quad_PTR[ww,:] = sw_quad_PR[ww,:]-(sw_quad_xZi+sw_quad_yZi)
    # ====================== #
#####################################################################
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

fig,axs = plt.subplots(figsize=(7,6),nrows=3,ncols=2,layout='constrained')

axs[0,0].semilogx(rhoD_quad/z*1/theta0[0],np.sqrt(sw_quad_PR[0,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[0,0].semilogx(rhoD/z*1/theta0[0],np.sqrt(PR_GI[0,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[0,0].semilogx(rhoD/z*1/theta0[0],np.sqrt(PR_GIV[0,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[0,0].set_xlabel(r'$\theta/\theta_0$')
axs[0,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[0,0].grid()
axs[0,0].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[0,:])/wvlT)*1.1])
axs[0,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[0]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[0]*1e6,1)}')
axs[0,0].text(0.003,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))
axs[0,0].legend(framealpha=1,fancybox=False,edgecolor='k', \
    bbox_to_anchor=(-0.25,1),ncol=1) 
   
axs[1,0].semilogx(rhoD_quad/z*1/theta0[1],np.sqrt(sw_quad_PR[1,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[1,0].semilogx(rhoD/z*1/theta0[1],np.sqrt(PR_GI[1,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[1,0].semilogx(rhoD/z*1/theta0[1],np.sqrt(PR_GIV[1,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[1,0].set_xlabel(r'$\theta/\theta_0$')
axs[1,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[1,0].grid()
axs[1,0].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[1,:])/wvlT)*1.1])
axs[1,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[1]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[1]*1e6,1)}')
axs[1,0].text(0.0015,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[2,0].semilogx(rhoD_quad/z*1/theta0[2],np.sqrt(sw_quad_PR[2,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[2,0].semilogx(rhoD/z*1/theta0[2],np.sqrt(PR_GI[2,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[2,0].semilogx(rhoD/z*1/theta0[2],np.sqrt(PR_GIV[2,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
axs[2,0].set_xlabel(r'$\theta/\theta_0$')
axs[2,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[2,0].grid()
axs[2,0].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[2,:])/wvlT)*1.1])
axs[2,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[2]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[2]*1e6,1)}')
axs[2,0].text(0.0012,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[0,1].semilogx(rhoD_quad/z*1/theta0[3],np.sqrt(sw_quad_PR[3,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[0,1].semilogx(rhoD/z*1/theta0[3],np.sqrt(PR_GI[3,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[0,1].semilogx(rhoD/z*1/theta0[3],np.sqrt(PR_GIV[3,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[0,1].set_xlabel(r'$\theta/\theta_0$')
#axs[0,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[0,1].grid()
axs[0,1].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[3,:])/wvlT)*1.1])
axs[0,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[3]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[3]*1e6,1)}')
axs[0,1].text(0.001,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[1,1].semilogx(rhoD_quad/z*1/theta0[4],np.sqrt(sw_quad_PR[4,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[1,1].semilogx(rhoD/z*1/theta0[4],np.sqrt(PR_GI[4,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[1,1].semilogx(rhoD/z*1/theta0[4],np.sqrt(PR_GIV[4,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[1,1].set_xlabel(r'$\theta/\theta_0$')
#axs[1,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[1,1].grid()
axs[1,1].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[4,:])/wvlT)*1.1])
axs[1,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[4]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[4]*1e6,1)}')
axs[1,1].text(0.001,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[2,1].semilogx(rhoD_quad/z*1/theta0[5],np.sqrt(sw_quad_PR[5,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[2,1].semilogx(rhoD/z*1/theta0[5],np.sqrt(PR_GI[5,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[2,1].semilogx(rhoD/z*1/theta0[5],np.sqrt(PR_GIV[5,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
axs[2,1].set_xlabel(r'$\theta/\theta_0$')
#axs[2,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[2,1].grid()
axs[2,1].set_ylim([-0.05,np.max(np.sqrt(sw_quad_PR[5,:])/wvlT)*1.1])
axs[2,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[5]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[5]*1e6,1)}')
axs[2,1].text(0.001,0.45,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

text1 = r'$\lambda_{\mathrm{T}}$ = {} $\mu$m'.replace('{}',f'{wvlT*1e6}')    
text2 = r'$C_n^2 = 5 \times 10^{-15}$ m$^{2/3}$'
text3 = r'$D$ = {} m'.replace('{}',f'{np.round(D,1)}')
text4 = r'$z$ = {} km'.replace('{}',f'{np.round(z/1e3,0)}')
plt.figtext(0,0.72,'\n'.join((text1,text2,text3,text4)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k')) 
       
plt.show()
fig.savefig('PR_w_aniso.pdf',bbox_inches='tight')

#####################################################################
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

fig,axs = plt.subplots(figsize=(7,6),nrows=3,ncols=2,layout='constrained')

axs[0,0].semilogx(rhoD_quad/z*1/theta0[0],np.sqrt(sw_quad_PTR[0,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[0,0].semilogx(rhoD/z*1/theta0[0],np.sqrt(PTR_GI[0,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[0,0].semilogx(rhoD/z*1/theta0[0],np.sqrt(PTR_GIV[0,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[0,0].set_xlabel(r'$\theta/\theta_0$')
axs[0,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[0,0].grid()
axs[0,0].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[0,:])/wvlT)*1.1])
axs[0,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[0]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[0]*1e6,1)}')
axs[0,0].text(0.003,0.2,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))
axs[0,0].legend(framealpha=1,fancybox=False,edgecolor='k', \
    bbox_to_anchor=(-0.25,1),ncol=1) 
   
axs[1,0].semilogx(rhoD_quad/z*1/theta0[1],np.sqrt(sw_quad_PTR[1,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[1,0].semilogx(rhoD/z*1/theta0[1],np.sqrt(PTR_GI[1,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[1,0].semilogx(rhoD/z*1/theta0[1],np.sqrt(PTR_GIV[1,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[1,0].set_xlabel(r'$\theta/\theta_0$')
axs[1,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[1,0].grid()
axs[1,0].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[1,:])/wvlT)*1.1])
axs[1,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[1]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[1]*1e6,1)}')
axs[1,0].text(0.0015,0.2,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[2,0].semilogx(rhoD_quad/z*1/theta0[2],np.sqrt(sw_quad_PTR[2,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[2,0].semilogx(rhoD/z*1/theta0[2],np.sqrt(PTR_GI[2,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[2,0].semilogx(rhoD/z*1/theta0[2],np.sqrt(PTR_GIV[2,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
axs[2,0].set_xlabel(r'$\theta/\theta_0$')
axs[2,0].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[2,0].grid()
axs[2,0].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[2,:])/wvlT)*1.1])
axs[2,0].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[2]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[2]*1e6,1)}')
axs[2,0].text(0.0012,0.2,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[0,1].semilogx(rhoD_quad/z*1/theta0[3],np.sqrt(sw_quad_PTR[3,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[0,1].semilogx(rhoD/z*1/theta0[3],np.sqrt(PTR_GI[3,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[0,1].semilogx(rhoD/z*1/theta0[3],np.sqrt(PTR_GIV[3,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[0,1].set_xlabel(r'$\theta/\theta_0$')
#axs[0,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[0,1].grid()
axs[0,1].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[3,:])/wvlT)*1.1])
axs[0,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[3]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[3]*1e6,1)}')
axs[0,1].text(0.001,0.195,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[1,1].semilogx(rhoD_quad/z*1/theta0[4],np.sqrt(sw_quad_PTR[4,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[1,1].semilogx(rhoD/z*1/theta0[4],np.sqrt(PTR_GI[4,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[1,1].semilogx(rhoD/z*1/theta0[4],np.sqrt(PTR_GIV[4,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
#axs[1,1].set_xlabel(r'$\theta/\theta_0$')
#axs[1,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[1,1].grid()
axs[1,1].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[4,:])/wvlT)*1.1])
axs[1,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[4]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[4]*1e6,1)}')
axs[1,1].text(0.0008,0.195,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

axs[2,1].semilogx(rhoD_quad/z*1/theta0[5],np.sqrt(sw_quad_PTR[5,:])/wvlT,'o',color='k',markerfacecolor='none',label=r'Quadrature')        
axs[2,1].semilogx(rhoD/z*1/theta0[5],np.sqrt(PTR_GI[5,:])/wvlT,'--',color='tab:orange',label=r'$\theta \gg \sqrt{\alpha_k}/z$')
axs[2,1].semilogx(rhoD/z*1/theta0[5],np.sqrt(PTR_GIV[5,:])/wvlT,'--',color='tab:green',label=r'$\theta \ll \sqrt{\alpha_k}/z$')
axs[2,1].set_xlabel(r'$\theta/\theta_0$')
#axs[2,1].set_ylabel(r'$\sqrt{{\langle \Delta \ell_{\mathrm{PTR}}^2 \rangle}}/\lambda_{{\mathrm{{T}}}}$')
axs[2,1].grid()
axs[2,1].set_ylim([-0.01,np.max(np.sqrt(sw_quad_PTR[5,:])/wvlT)*1.1])
axs[2,1].set_xticks([0.001,0.01,0.1,1,10,100,1000])
text1 = r'$\lambda_{\mathrm{B}}$ = {} $\mu$m'.replace('{}',f'{np.round(wvlB[5]*1e6,1)}')
text2 = r'$\theta_0$ = {} $\mu$rad'.replace('{}',f'{np.round(theta0[5]*1e6,1)}')
axs[2,1].text(0.0008,0.195,'\n'.join((text1,text2)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k'))

text1 = r'$\lambda_{\mathrm{T}}$ = {} $\mu$m'.replace('{}',f'{wvlT*1e6}')    
text2 = r'$C_n^2 = 5 \times 10^{-15}$ m$^{2/3}$'
text3 = r'$D$ = {} m'.replace('{}',f'{np.round(D,1)}')
text4 = r'$z$ = {} km'.replace('{}',f'{np.round(z/1e3,0)}')
plt.figtext(0,0.72,'\n'.join((text1,text2,text3,text4)),fontsize=10, \
    backgroundcolor='w',bbox=dict(facecolor='w', edgecolor='k')) 
       
plt.show()
fig.savefig('PTR_w_aniso.pdf',bbox_inches='tight')
