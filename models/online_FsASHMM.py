# -*- coding: utf-8 -*-
"""
@date: Created on Thu Oct 29 14:35:32 2020
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@email: epuerto@ainguraiiot.com
@Licence: CC BY-NC-ND 4.0
"""
import pandas as pd
from AsFS_AR_ASLG_HMM import AsFS_AR_ASLG_HMM as hmm
from FS_AR_ASLG_HMM import FS_AR_ASLG_HMM as fhmm
import warnings
warnings.filterwarnings("ignore")
import math
import datetime
import os
import pickle
import numpy as np
import matplotlib.pyplot as  plt
import networkx as nx
from  scipy.signal import hilbert
class data_stream:
    def __init__(self,x,L,slide,dt,rho_bar=0.9,P=5,p = None, pp=0.1,alpha=3,gamma2=0.05,tt=0,div = "cs", mod=None,
                 nu=None,kappa=None,mode="max",lags=True,struc=True,hang = True,old=False,plot_health= True,
                 thr_div= 0.0,levels=[-0.25,-0.5,-0.75,-1,-1.25,-1.5,-1.75,-2,-2.25,-2.5],labels=None,asym = True,Lmax =None, RUL_comp=True):
        # Models parameters}
        if Lmax == None:
            self.Lmax = L+1
        else:
            if Lmax<L:
                self.Lmax = L+1
            else:
                self.Lmax=Lmax
        self.asym = asym
        self.rho_bar=rho_bar
        self.plot_health = plot_health 
        self.hang = hang   # Hay olgura para determinar un cmabio
        self.div = div     # Tipo de divergencia a utilizar para la diveregncia entre estados ocultos
        self.old = old     # Pregunta si usa evolución de estados ocultos
        self.struc = struc # Realizar optimizacion de estructura
        self.lags = lags   # Realizar optimización de rezagos
        self.x = x         # Datos con los cuales se emulara el data stream
        self.T = len(x)    # Longitud total de los datos
        self.P = P         # Maximo valor de autoregresion permisible 
        self.K = len(x[0]) # Numero de variables
        self.colores = self.create_far_colors(self.K)
        self.L = L         # Longitud de la ventana a trabajar
        self.div_mat = None# Divergence Matrix
        self.dt = dt       # Tiempo entre observaciones
        if nu is None:
            self.nu = np.ones(self.K) # Valores de referencia estados ocultos
        else:
            self.nu =nu  
        if kappa is None:
            self.kappa = np.zeros(self.K) # Escalados de variables de estados ocultos
        else:
            self.kappa = kappa
        if labels == None:
            self.labels= []
            for k in range(self.K):
                self.labels.append(r"$X_{"+str(k+1)+"}$")
        else:
            self.labels= labels
            
        # Window size selection for concept drift
        self.it = 1
        self.pp = pp                        # porcentaje de anomalias admisibles
        self.pht = []                       # evolución del test de Page
        self.drift = []                     # Binary variale which indicates when a drift has been done
        self.slide = slide                  # Cantidad de datos que se desplaza la ventana 
        # self.gamma = L*np.log(alpha)        # Parametro de control del test de Page
        self.gamma = np.log(alpha)
        self.gamma2 = gamma2                # Parametro de control del test bernoulli, depende de L
        if hang == True:
            self.label = np.zeros(self.min_sample(self.gamma2, pp, 1e-2)) #Determina que datos de la ventana son anomalias, 0 es normal, 1 es anomalia.
        else:
            self.label = np.zeros(1)
        self.Sn= []              # Parametro del page-Hinkey
        self.mean = 0            # Media actual de la verosimilitud encontrada
        
        # Definicion del modelo
        if mod == None:
            self.N = 1
            if self.asym == True:
                self.model = hmm(x[:L],np.array([L]),self.N,P=P,lags=lags,struc=struc,rho_bar=rho_bar,act_inde=False) #Definicion del modelo
                self.model.label_parameters(self.kappa,self.nu) # Indicar los paramaetros de escalado para el algoritmo Viterbi
                self.viterbi = np.array([0]) # Sucesion de viterbi
                self.firstt = True
            else:
                self.model = fhmm(x[:L],np.array([L]),self.N,P=P,lags=lags,struc=struc,rho_bar=rho_bar,act_inde=False) #Definicion del modelo
                self.model.label_parameters(self.kappa,self.nu) # Indicar los paramaetros de escalado para el algoritmo Viterbi
                self.viterbi = np.array([0]) # Sucesion de viterbi
                self.firstt = True
        else:
            self.N =mod.N
            if self.asym == True:
                self.model = mod
                self.model.label_parameters(self.kappa,self.nu) # Indicar los paramaetros de escalado para el algoritmo Viterbi
                self.viterbi = np.array([0]) # Sucesion de viterbi
                self.firstt = False
            else:
                self.model = mod
                self.model.label_parameters(self.kappa,self.nu) # Indicar los paramaetros de escalado para el algoritmo Viterbi
                self.viterbi = np.array([0]) # Sucesion de viterbi
                self.firstt = False
        # Para pintar
        self.s_reg = []              # Evolución desviación estandar predicción
        self.rhot = []               # Evolución de relevancia de 
        self.bict = []               # BIC computados hasta el momento
        self.hi_mu = []              # Parametros de salud base de la media
        self.hi_cov= []              # Parametros de salud base de la varianza
        self.health = []             # Salud del rodamiento
        self.healthk = []            # Salud de diferentes partes del rodamiento
        self.tt =tt                  # Horizonte de prediccion
        self.RUL = []                # RUL prediction
        self.tRUL =[]                # Eje de tiempo para RUL
        self.mtetha =  []            # Variacion temporal de la pendiente de la linea tangente
        self.btetha = []             # Variacion temporal del interecepto de la linea tangente
        self.meanhi1 = 0             # 
        self.t_drift =[]             # Eje de tiempo para drift
        self.f= []                   # Timepo remanente par allegar a diferentes niveles de degradacion
        self.mf = 1
        self.levels = levels
        self.mode = mode
        self.div_health = []
        self.RUL_div = []
        self.ps = -1
        self.thr_div= thr_div
        self.RUL_compute  = RUL_comp
        self.pred = None
        self.r_sq = None
        self.sigmar2 = None
        
    def create_far_colors(self,n,plot=False):
        """
        """
        def z_rotation(vector,theta):
            """Rotates 3-D vector around z-axis"""
            R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
            return np.dot(R,vector)
        colors = []
        from itertools import product
        bounds = np.array([0.9,0])
        count = 3
        perm = list(product([0,1],[0,1],[0,1]))
        seguir = True
        rot = 0
        while seguir:
            for i in perm:
                colori = bounds[list(i)]
                if rot ==1:
                    colori = z_rotation(colori-np.ones(3)*0.5,np.pi/4)+np.ones(3)*0.5
                colors.append(colori)
                if len(colors) ==n+1:
                    seguir = False
                    break
            if rot ==0:
                rot = 1
            else:
                rot=  0
            bounds = np.array([0.5-1/count,0.5+1/count])
            count = count*2
        colors = np.array(colors)    
        if plot==True:
            fig = plt.figure("colors")
            fig.clf()
            ax = fig.add_subplot(projection='3d')
            for i in range(n):
                ax.scatter(colors[i,0],colors[i,1],colors[i,2],color= colors[i])
            ax.grid("on")
        return colors[1:]
         
    def train_model(self,its1,its2):
        """
        Trains the model
        """
        # self.model.EM()
        O =self.model.O
        self.model.SEM(its1=its1,its2=its2)
        if (np.min(self.model.sigma)<1e-10) or np.isnan(self.model.LogLtrain):
            print("############# small variances problem detected in SEM  or NaN values appeared #####################")
            if self.asym == True:
                self.model = hmm(O,np.array([len(O)]),1, struc=self.struc,lags=self.lags,rho_bar=self.rho_bar,act_inde=False)
            else:
                self.model = fhmm(O,np.array([len(O)]),1, struc=self.struc,lags=self.lags,rho_bar=self.rho_bar,act_inde=False)   
            self.model.label_parameters(self.kappa,self.nu)
            self.model.EM(its =its2) 
        
    def train_model_run(self,x,its1,its2,ps=None):
        """
        """
        leng =float( len(x))
        epsi = self.model.epsi
        tau = self.model.tau
        B = self.model.B
        A = self.model.A
        pi = self.model.pi
        S = self.model.sigma
        G = self.model.G
        L = self.model.L
        p = self.model.p
        rho = self.model.rho
        [pi,A,B,S,L,G,p,rho] = self.add_new_state(pi,A,B,S,L,G,p,rho,x,self.N)
        ta = np.zeros(self.N+1)
        tb = np.zeros(self.N+1)
        ts = np.zeros(self.N+1)
        ta[self.N]   = 1
        ts[self.N]   = 1
        tb[self.N]   = 1
        if self.asym == True:
            model2 = hmm(x,np.array([len(x)]), self.N+1,A=A,pi=pi,B=B,sigma=S, G=G,L=L,p=p,struc=self.struc,lags=self.lags,rho=rho,rho_bar=self.rho_bar,epsi=epsi,tau=tau,act_inde=False)
        else:
            model2 = fhmm(x,np.array([len(x)]), self.N+1,A=A,pi=pi,B=B,sigma=S, G=G,L=L,p=p,struc=self.struc,lags=self.lags,rho=rho,rho_bar=self.rho_bar,epsi=epsi,tau=tau,act_inde=False)   
        model2.label_parameters(self.kappa,self.nu)
        model2.SEM(its1=its1,its2=its2,ta=ta,tb=tb,ts = ts, tpi=False,ps=ps)
        if np.isnan(model2.LogLtrain):
            print("############# nan values problem detected in SEM #####################")
            if self.asym == True:                
                model2 = hmm(x,np.array([len(x)]), self.N+1,A=A,pi=pi,B=B,sigma=S, G=G,L=L,p=p,struc=self.struc,lags=self.lags,rho=rho,rho_bar=self.rho_bar,epsi=epsi,tau=tau,act_inde=False)
            else:
                model2 = fhmm(x,np.array([len(x)]), self.N+1,A=A,pi=pi,B=B,sigma=S, G=G,L=L,p=p,struc=self.struc,lags=self.lags,rho=rho,rho_bar=self.rho_bar,epsi=epsi,tau=tau,act_inde=False)   
            model2.label_parameters(self.kappa,self.nu)
            model2.EM(its =its2,ta=ta,tb=tb,ts = ts, tpi=False,ps=ps)
            if np.isnan(model2.LogLtrain):
                print("############# nan values problem detected in EM  #############")           
        else:
            if math.isnan(model2.log_likelihood(x,pena=True,ps=ps)):
                print("No se aprendio nada")
                return None
            elif self.model.log_likelihood(x,pena=True,ps=ps)> model2.log_likelihood(x,pena=True,ps=ps) :
            # if True:
                t_real = len(self.health)*self.dt
                self.t_drift.append(t_real)
                self.model=model2
                if self.hang == True:
                    self.label = np.zeros(self.min_sample(self.gamma2,self.pp, 1e-2)) #Determina que datos de la ventana son anomalias, 0 es normal, 1 es anomalia.
                else:
                    self.label = np.zeros(1)
                self.Sn = [self.model.bic/leng]
                if self.old== False:
                    self.N= self.N+1
                self.mean = self.model.bic/leng
                self.it = 1
                # pol = poly()
                # pol.plot_AR(model2.B, model2.p, model2.N, model2.K, model2.dictionary)
                # # Actualizar indices de salud base si se detecta mejora en alguna variable
                # minmu = np.argmin(np.sum(self.model.mus,axis=1))
                # minvar = np.argmin(np.sum(np.sum(np.abs(self.model.covs),axis=1),axis=1)) 
                # mhi_mu   = np.copy(self.model.mus[minmu])
                # mhi_cov = np.copy(np.diag(self.model.covs[minvar]))
                self.div_mat =self.divergence("cs")
                # for k in range(self.K):
                #     if mhi_mu[k] < self.hi_mu[k] and  not math.isnan(mhi_mu[k]):
                #         self.hi_mu[k] = mhi_mu[k]
                #     if mhi_cov[k] < self.hi_cov[k] and not math.isnan(mhi_cov[k]):
                #         self.hi_cov[k] = mhi_cov[k]
                # print("$\mu_h2$: " +str(mhi_mu))
                # print("$\sigma^2_h2$: " +str(mhi_cov))
                for k in range(self.K):
                    for n in range(self.N):
                        if self.hi_mu[k]>self.model.mus[n,k]:
                            self.hi_mu[k] = self.model.mus[n,k]
                        if self.hi_cov[k]>(self.model.covs[n])[k,k]:
                            self.hi_cov[k] =(self.model.covs[n])[k,k]
            else:
                print("############# poor fitness in new model #####################")
            
    def add_new_state(self,pi,A,B,S,L,G,p,rho,x,N):
        """
        Generates a set pf áramete
        Parameters
        ----------
        pi : 1D-array
            Initial distribution of hidden states
        A  : TYPE 2D-array
            Transition Matrix
        B  : TYPE list
            Emission probabilities
        S  : list
            Variances
        L  : list
            Topological orders
        G  : list
            Graphs
        p  : 2D-array
            AR orders 
        Returns A new set of parameters with a new hidden state
        """
        Gn = list(np.copy(G))
        Gn.append(np.zeros([self.K,self.K]))
        Ln = list(np.copy(L))
        Ln.append(np.arange(self.K))
        pin = np.concatenate([pi,[0]])
        pn = np.copy(p)
        if self.lags == True:
            pn = np.concatenate([pn,np.zeros([1,self.K])],axis=0).astype(int)
        if self.lags == False:
            pn = np.concatenate([pn,self.P*np.ones([1,self.K])],axis=0).astype(int)
        # Bn = list(np.copy(B))
        Bn = []
        for i in range(self.N):
            Bni = []
            for k in range(self.K):
                bn = np.copy(B[i][k])
                Bni.append(bn)
            Bn.append(Bni)
        Sn = list(np.copy(S))
        Bi = []
        sigmai =[]
        for k in range(self.K):
            # sigmaik = np.abs(np.min(x.T[k])-np.max(x.T[k]))
            sigmaik = np.var(x[-self.L:,k])
            # Bik = np.concatenate([[np.mean(x[:,k])],np.zeros(pn[N][k])])
            Bik = np.concatenate([[np.median(x[-self.L:,k])],np.zeros(pn[N][k])])
            Bi.append(Bik)
            sigmai.append(sigmaik)
        Bn.append(np.array(Bi))
        Sn = np.concatenate([Sn,np.array(sigmai)[np.newaxis]],axis=0)
        An = np.copy(A)
        An = np.concatenate([An,np.zeros([N,1])],axis=1)
        An = np.concatenate([An,np.ones([1,N+1])],axis=0)
        # An[N-1][N] = 1e-6
        An[:,N] = 1e-6
        for i in range(N+1):
            An[i] = An[i]/np.sum(An[i])
        if self.asym == True:
            rhon = np.concatenate([rho,0.1*np.ones([1,self.K])],axis=0)
        else:
            rhon= rho
        return [pin,An,Bn,Sn,Ln,Gn,pn,rhon]
            
    def page_hinkley(self,t,ot): # TODO falta revisarlo
        """
        Realiza el test de Page-Hinkley para una variable, returna True si se 
        detecta un cambio significativo, false de lo contrario
        """
        nmu = ot/t+(t-1)/t*self.mean 
        S = nmu
        self.Sn.append(S)
        self.mean = nmu
        Ph = S-np.min(np.array(self.Sn))
        self.pht.append(Ph)
        print("Page hinkley threshold: " + str(Ph))
        if Ph > self.gamma:
            return 1
        else:
            return 0

        
    def plot_page_hinkley(self):
        times = self.L*self.dt+self.slide*np.arange(len(self.pht))*self.dt
        times = np.array(times)
        plt.figure("Page test FS",figsize=(4.5,3))
        plt.clf()
        plt.grid("on")
        plt.plot(times,self.pht)
        plt.plot(times,self.gamma*np.ones(len(times)),c="green")
        # plt.plot(times,20*self.gamma*np.ones(len(times)),c="red")
        plt.xlabel("Time [h]")
        plt.ylabel("Page test")
        plt.xlim(left=0)
        plt.tight_layout()
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.pause(0.02)
        
                
    def bernoulli_kullback_leibler(self,p,q): 
        """
        Determina la divergencia de Kullback-Leibler de dos distribuciones Bernoulli 
        D(p|q) = qlog(q/p)+(1-q)log((1-q)/(1-p))
        """
        return q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))
    
    def min_sample(self,delta,p,epsilon):
        """
        Queremos encontrar un n lo suficiente pequeño tal que:
            P(|p-p'|<e)> 1-delta
            p' = sum(x_i)/n donde x_i es bernoulli con parametro p
            asumimos que p>p' , o que el parametro p real es mayor al estimado antes de detectar el cambio de concepto
        Usamos el hecho que:
            P(p'>p-e) < exp(-D((p-e)|p)n)
        Por lo tanto, se tiene que:
            -log((1-delta))/D((p-e)|p)>n
        O en otras palabras, si n es mayor o igual a la cota, se cumple la condicion deseada
        """
        n= -np.log(1-delta)/self.bernoulli_kullback_leibler(p-epsilon, p)
        print("Size of outliers window: "+ str(n))
        return int(np.ceil(n))
    
            
    
    def test_bernoulli(self,labels): 
        """
        Determina si la proporcion de anomalias es importante en la ventana de tiempo tomada
        En caso de ser positivo, se considera un posible concept drift.
        """
        phat = np.mean(np.array(labels))
        print("Current p_hat= " +str(phat))
        if phat>self.pp:
            return True
        else:
            return False
        
    def base_health_index(self):
        """
        Defines the health index 
        """
        mincov = np.argmin(np.sum(np.sum(np.abs(self.model.covs),axis=1),axis=1)) 
        minmu = np.argmin(np.sum(np.abs(self.model.mus),axis=1))
        self.hi_mu   = np.copy(self.model.mus[minmu])
        self.hi_cov = np.copy(np.diag(self.model.covs[mincov]))
        for k in range(self.K):
            for n in range(self.N):
                if self.hi_mu[k]>self.model.mus[n,k]:
                    self.hi_mu[k] = self.model.mus[n,k]
                if self.hi_cov[k]>(self.model.covs[n])[k,k]:
                    self.hi_cov[k] =(self.model.covs[n])[k,k]
        print("$\mu_h$: " +str(self.hi_mu))
        print("$\sigma^2_h$: " +str(self.hi_cov))
        
        
         
         
    def health_index_init(self,x_t,mode="sum"):
        """
        Computa los estados de salud del nuevo conjunto de datos
        """
        indess = self.model.viterbi(x_t,indexes=True,plot=False)
        healts = []
        for t in range(len(x_t)-self.P):
            health_t = []
            for k in range(self.K):
                i = indess[t]
                healthtkmu = -np.log(np.abs(self.model.mus[i][k])/self.hi_mu[k])
                healthtksigma = -np.log(np.sqrt(np.abs(self.model.covs[i][k][k])/self.hi_cov[k]))
                health_t.append(healthtkmu)
                health_t.append(healthtksigma)
            healts.append(health_t)
        HIs = np.array(healts)
        if mode=="mean":
            gHI = np.mean(HIs,axis=1)
        if mode =="max":
            gHI = np.min(HIs,axis=1)
        gHI = gHI[np.newaxis].T
        return [HIs,gHI]
    
            
    def health_index_new(self,x_t,mode="sum"):
        """
        Computa los estados de salud del nuevo conjunto de datos
        usando la funcion:
        """
        indess = self.model.viterbi(x_t,indexes=True,plot=False,ps = self.ps)
        
        healts = []
        for t in range(len(indess)-self.slide,len(indess)):
            health_t = []
            for k in range(self.K):
                i = indess[t]
                if self.asym== True:
                    if self.model.rho[i][k]>self.rho_bar or self.N==1:
                        healthtkmu = -np.log(np.abs(self.model.mus[i][k])/self.hi_mu[k])
                        healthtksigma = -np.log(np.sqrt(np.abs(self.model.covs[i][k][k])/self.hi_cov[k]))
                        health_t.append(healthtkmu)
                        health_t.append(healthtksigma)
                    else:
                        healthtkmu = -np.log(np.abs(self.model.epsi[k])/self.hi_mu[k])
                        healthtksigma = -np.log(np.sqrt(np.abs(self.model.tau[k])/self.hi_cov[k]))
                        health_t.append(healthtkmu)
                        health_t.append(healthtksigma)         
                else:
                    if self.model.rho[k]>self.rho_bar or self.N==1:
                        healthtkmu = -np.log(np.abs(self.model.mus[i][k])/self.hi_mu[k])
                        healthtksigma = -np.log(np.sqrt(np.abs(self.model.covs[i][k][k])/self.hi_cov[k]))
                        health_t.append(healthtkmu)
                        health_t.append(healthtksigma)
                    else:
                        healthtkmu = -np.log(np.abs(self.model.epsi[k])/self.hi_mu[k])
                        healthtksigma = -np.log(np.sqrt(np.abs(self.model.tau[k])/self.hi_cov[k]))
                        health_t.append(healthtkmu)
                        health_t.append(healthtksigma)       
            healts.append(health_t)
        HIs = np.array(healts)
        if mode=="mean":
            gHI = np.mean(HIs,axis=1)
        if mode =="max":
            gHI = np.min(HIs,axis=1)
        gHI = gHI[np.newaxis].T
        return [HIs,gHI]
    
    def refresh_health(self):
        """
        Actualiza los plots de salud y de la lectura de viterbi
        """
        health = np.array(self.health)
        healthk  = np.array(self.healthk)
        plt.figure("Health",figsize=(4.5,3))
        plt.clf()
        plt.subplots_adjust(hspace = 0.4)
        plt.subplots_adjust(wspace = 0.6)
        for i in range(self.K+1):
            if i <self.K:
                plt.subplot(self.K+1,1,i+1)
                plt.title("Variable "+str(i)+" health index")
                plt.plot(healthk.T[i])
                plt.grid("on")
                plt.ylabel("Health index")
                plt.xlabel("Time[hours]")
                plt.pause(0.02)
            else:
                plt.subplot(self.K+1,1,i+1)
                plt.title("Global health index")
                plt.plot(health)
                plt.grid("on")
                plt.ylabel("Health index")
                plt.xlabel("Time[hours]")
                plt.pause(0.02)
                
        
                
    
    def det_model(self):
        """
        regresión cuadratica pues
        """
        hel = np.copy(self.health).reshape([len(self.health),1])
        
        times = np.arange(len(self.health))*self.dt+self.dt
        times = times[np.newaxis].T
        
        timesn = np.copy(times)/np.max(times)
        times2 = timesn**2
        timest = np.concatenate([np.ones([len(timesn),1]),timesn,times2],axis=1)
        model2 = linear_reg(timest,hel,constant=False)
        model2.fit()
        pred = np.dot(timest,model2.beta)
        return [model2,pred]
    
    
        
    def HI_pred(self,model,nm,t):
        """
        Predicts the Following self.tt health index
        """
        c = np.max(np.arange(len(self.health))*self.dt+self.dt)
        timesf = np.arange(len(self.health))*self.dt+self.dt+np.min([t,self.tt])*self.dt
        timesfn = timesf[np.newaxis].T/c
        timesf2 = timesfn**2
        timesF = np.concatenate([timesfn,timesf2],axis=1)
        if nm!= 0:
            pred  = model.pred(timesF)
        else:
            pred  = model.pred(timesfn)
        return pred
    
    
        
    def RUL_levels(self,reg_model,times):
        """
        jaja que chistoso soy
        """
        f_times = []
        for j in range(len(self.levels)):
            mpol = np.copy(reg_model.beta[::-1])
            mpol[-1] = mpol[-1]-self.levels[j]
            mpol = mpol+0j
            tf1=(-mpol[1]+np.sqrt(mpol[1]**2-4*mpol[0]*mpol[2]))/(2.*mpol[0])
            tf2 = (-mpol[1]-np.sqrt(mpol[1]**2-4*mpol[0]*mpol[2]))/(2.*mpol[0])
            tf = max(tf1,tf2)
            # tff = poly().all_roots(mpol)[0]
            # tf = np.max(np.real(tff))
            tf = tf*np.max(times)
            if np.real(tf) > times[-1] and np.abs(np.imag(tf)) <1e-10 and mpol[0]<0 :
                f_times.append((np.real(tf)-times[-1])[0])
            else:
                f_times.append(0.)
                
        if np.max(np.array(f_times)) >self.mf :
            self.mf = np.max(f_times)
        self.f.append(f_times)
        if self.plot_health == True:
            plt.figure("Table of remaining condition time fs",figsize=(4.5,3))
            plt.clf()
            plt.grid("on")
            plt.xlim(left=0, right=-np.min(self.levels)+0.2)
            plt.xlabel("Degradation levels")
            plt.ylabel("Remaining hours [h]")
            nl = len(self.levels)
            for i in range(len(self.levels)):
                gr = float(i)/(float(nl)-1)
                plt.bar(-self.levels[i],f_times[i],width =0.125, color = (gr,(1-gr),0))
            plt.tight_layout()
            plt.pause(0.02)
        
    def predict_RUL(self):
        """
        Predice el RUL
        """
        [reg_model,pred] = self.det_model()
        pred = pred[:,0]
        sigmar2 = reg_model.sigma
        self.s_reg.append(sigmar2)
        r_sq = reg_model.R2bar
        times = np.arange(len(self.health))*self.dt+self.dt
        times = times[np.newaxis].T
        self.RUL_levels(reg_model,times)
        
        mpol      = np.copy((reg_model.beta.T[0])[::-1])
        mpol[-1]  = mpol[-1]-self.levels[-1]
        mpol      = mpol+0j
        tf1       = (-mpol[1]+np.sqrt(mpol[1]**2-4*mpol[0]*mpol[2]))/(2.*mpol[0])
        tf2       = (-mpol[1]-np.sqrt(mpol[1]**2-4*mpol[0]*mpol[2]))/(2.*mpol[0])
        tf        = max(tf1,tf2)
        tf        = tf*np.max(times)
        
        
        if np.real(tf) > times[-1] and np.abs(np.imag(tf)) <1e-10 and mpol[0]<0  :
            self.RUL.append((np.real(tf)-times[-1])[0])
        else:
            self.RUL.append(math.nan)
            
        self.tRUL.append(times[-1])
        self.pred = pred
        self.r_sq = r_sq
        self.sigmar2 = sigmar2
        
    def plot_RUL(self):

        plt.figure("RUL_FS",figsize=(4.5,3))
        plt.clf()
        # plt.title("RUL")
        plt.plot(self.tRUL,self.RUL, label = "Current Health", color="blue") 
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylim(bottom=0)
        plt.grid("on")
        plt.xlabel("Time [h]")
        plt.ylabel("RUL[h]")
        plt.xlim(left=0)
        plt.tight_layout()
        plt.pause(0.02)
        
        plt.figure("reg_std",figsize=(4.5,3))
        plt.clf()
        # plt.title("RUL")
        plt.plot(self.tRUL,self.s_reg, label = "Current Health", color="blue") 
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylim(bottom=0)
        plt.grid("on")
        plt.ylabel("$\sigma_{HI^\dagger}$ ")
        plt.xlabel("Time[h]")
        plt.xlim(left=0)
        plt.tight_layout()
        plt.pause(0.02)
        
        cmax = np.abs(min(1,self.pred[-1]/(1*self.levels[-1])))

        plt.figure("Health index prediction FS",figsize=(4.5,3))
        plt.clf()
        plt.grid("on")
        if self.model.N>1:
            plt.plot(np.arange(len(self.pred))*self.dt,self.pred,label = "Fitted  $R^2$: "+ str(round(self.r_sq,2)),linewidth=5,color=(cmax,1.-cmax,0))
            plt.fill_between(np.arange(len(self.pred))*self.dt, self.pred - 1.96*self.sigmar2, self.pred + 1.96*self.sigmar2, alpha=0.2,color=(cmax,1.-cmax,0))
        plt.plot(np.arange(len(self.pred))*self.dt,self.health,label = "Health",linestyle = ":",linewidth=3) #toca rreglar el eje del tiempo
        plt.ylim(bottom=np.min(self.levels)-0.1)
        plt.ylim(top=0.2)
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylabel(r"$HI^{\dagger}$")
        plt.xlabel("Time [h]")
        plt.legend(loc =3)
        plt.xlim(left=0)
        plt.tight_layout()
        plt.pause(0.002)
        
        plt.figure("Health indexes fs",figsize=(4.5,3))
        plt.clf()
        plt.grid("on")
        for k in range(len(self.healthk[0])):
            if np.mod(k,2) ==0:
                plt.plot(np.arange(len(self.healthk))*self.dt,self.healthk[:,k],label = r"$HI_{\mu}$"+ " "+ self.labels[int(k/2)],color=self.colores[int(k/2)],linestyle = ":",linewidth=2)
            else:
                plt.plot(np.arange(len(self.healthk))*self.dt,self.healthk[:,k],label = r"$HI_{\Sigma}$"+ " "+ self.labels[int((k-1)/2)],color=self.colores[int((k-1)/2)],linestyle = "--",linewidth=2)
        plt.ylim(bottom=np.min(self.levels)-0.1)
        plt.ylim(top=0.2)
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylabel("$HI^{\dagger}$")
        plt.xlabel("Time [h]")
        plt.legend(loc =3)
        plt.xlim(left=0)
        plt.tight_layout()
        plt.pause(0.002)
            
    def plot_fs_evolution(self):
        """
        Plot the feature saliencies
        """
        rows = np.ceil(self.K/2)
        plt.figure("Relevancy evolution FS",figsize=(4.5,3))
        plt.clf()
        for k in range(self.K):   
            plt.subplot(rows,2,k+1)
            plt.plot(self.dt*np.arange(len(self.rhot)),np.array(self.rhot)[:,k],linestyle = "solid",label = self.labels[k],linewidth=1.5,color=self.colores[k])
            plt.ylim(bottom=-0.05)
            plt.ylim(top=1.05)
            for i in range(len(self.t_drift)):
                plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            plt.legend(loc =3)
            plt.xlim(left=0)
            plt.grid("on")
        plt.tight_layout()
        plt.pause(0.002)
        
    def plot_fs_evolution_particular(self,k):
        plt.figure("Relevancy evolution FS"+ str(k),figsize=(4.5,3))
        plt.clf()
        plt.plot(self.dt*np.arange(len(self.rhot)),self.rhot[:,k],linestyle = "solid",label = self.labels[k],linewidth=1.5,color=self.colores[k])
        plt.ylim(bottom=-0.05)
        plt.ylim(top=1.05)
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylabel(r"$\rho$")
        plt.xlabel("Time [h]")
        plt.legend(loc =3)
        plt.xlim(left=0)
        plt.grid("on")
        plt.tight_layout()
        
        
        
    def plot_hidden_states(self):
        plt.figure("Hidden states FS",figsize=(4.5,3))
        plt.clf()
        plt.grid("on")
        plt.plot(self.dt*np.arange(len(self.viterbi_hist)),self.viterbi_hist,linestyle = "solid",linewidth=1.5)
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.ylabel(r"$g$")
        plt.xlabel("Time [h]")
        plt.xlim(left=0)
        plt.tight_layout()
        plt.pause(0.002)
        
    
    def plot_measures(self):
        if self.L+self.t*self.slide<len(self.x):
            xt = self.x[:self.L+self.t*self.slide]
        if self.L+self.t*self.slide>len(self.x):
            xt = self.x
        
        rows = np.ceil(self.K/2)
        plt.figure("Measures fs",figsize=(4.5,3))
        plt.clf()
        for k in range(self.K):
            plt.subplot(rows,2,k+1)
            plt.plot(np.arange(len(xt))*self.dt,xt[:,k],color=self.colores[k],label=self.labels[k])
            for l in range(len(self.t_drift)):
                plt.axvline(x=self.t_drift[l],color="black",linestyle="--")
            plt.grid("on")
            plt.xlabel("hours [h]")
            plt.ylabel(r"$g$")
            plt.legend(loc=1)
            plt.tight_layout()
        plt.pause(0.002)
        
    def plot_measures_k(self,k):
        plt.figure("Measures fs variable"+str(k),figsize=(4.5,3))
        plt.clf()
        plt.plot(np.arange(int(self.L+(self.t-1)*self.slide))*self.dt,self.x[:int(self.L+(self.t-1)*self.slide),k],color=self.colores[k],label=self.labels[k])
        for l in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[l],color="black",linestyle="--")
        plt.grid("on")
        plt.xlabel("hours [h]")
        plt.ylabel(r"$g$")
        plt.legend(loc=1)
        plt.tight_layout()
        plt.pause(0.002)
        
    
    def refresh_bic(self):
        """
        Plot the BIC
        """
        times = self.L*self.dt+(np.arange(len(self.bict)))*self.slide*self.dt
        plt.figure("BIC FS", figsize=(4.5,3))
        plt.clf()
        plt.ylabel("BIC")
        plt.xlabel("Time [h]")
        plt.plot(times,self.bict)
        for i in range(len(self.t_drift)):
            plt.axvline(x=self.t_drift[i],color="black",linestyle="--")
        plt.grid("on")
        plt.tight_layout()
        plt.xlim(left=0)
        plt.pause(0.02)
        
    def low_pass_filter(self,x,HB1=0.05,HB2=0.1):
        """
        """
        xf = np.fft.fft(x)
        lent = len(x)
        filt = np.zeros(lent)
        for i in range(int(lent/2.0)):
            if i < lent*HB1:
                filt[i] = 1.
                filt[lent-i-1] = filt[i]
            if i >=lent*HB1 and i < lent*(HB1+HB2):
                filt[i] =1 - 0.5*(1.-np.cos(np.pi*(i-lent*HB1)/(HB2*lent)))
                filt[lent-i-1] = filt[i]
        return  np.real(np.fft.ifft(xf*filt))
    
    def movmean(self, x_array, window_size=10):
        xdt = pd.DataFrame({"x": x_array})
        x_me = xdt.rolling(window_size, min_periods=1, center=True).mean()
        x_men = np.array(x_me)
        return x_men.reshape(len(x_men),)
    
    def quad(self,x1,A,x2):
        """
        Quadratic form x_1.TAx_2
        A is a matrix, x1 a vector and x2 another vector
        """
        return np.dot(np.dot(x1,A),x2)
    
    def entropy_gauss(self,mu,sigma):
        """
        Entropy of a multi variate Gaussian distribution
        """
        return 0.5*np.log(np.linalg.det(2*np.pi*np.e*sigma))

    def kl_div(self,mu,sigma,nu,tao):
        """
        Computes the Kullback-Liebler divergence between two multivariate 
        Gaussian distributions
        the distance from D_1 ~ N(mu,sigma) to D_2 ~ N(nu,tao) or KL(D_1|D_2)
        """
        d = len(mu)
        det1 = np.linalg.det(sigma)
        det2 = np.linalg.det(tao)
        inv2 = np.linalg.inv(tao)
        t1 = np.log(np.abs(det1)/np.abs(det2))-d
        t2 = np.sum(np.diag(np.dot(inv2,sigma)))
        t3 = self.quad(mu-nu,inv2,mu-nu)
        return 0.5*(t1+t2+t3)
    
    def cs_div(self,mu,sigma,nu,tao):
        """
        Cauchy-Schwarz divergence for two normals
        """
        d = float(len(mu))
        p1 = 0.5*np.log(np.abs(np.linalg.det(sigma+tao))*(2*np.pi)**d)
        p2 = 0.5*self.quad(mu-nu,np.linalg.inv(sigma+tao),mu-nu)
        p3 = -0.25*np.log(np.abs(np.linalg.det(sigma)*np.linalg.det(tao))*(4*np.pi)**(2*d)) 
        return p1+p2+p3

        
    def zpq(self,mu,nu,sigma,tao):
        """
        Para el calculo de Cauchy-Schwarz 
        """
        d = float(len(mu))
        # isig = np.linalg.inv(sigma)
        # itau = np.linalg.inv(tao)
        # detm = np.linalg.det(isig+itau)
        # invm = np.linalg.inv(isig+itau)
        # prob = (2*np.pi)**(-d*0.5)*np.abs(detm)**(-0.5)*np.exp(-0.5*self.quad(mu-nu,invm,mu-nu))
        detm = np.linalg.det(sigma+tao)
        invm = np.linalg.inv(sigma+tao)
        prob = (2*np.pi)**(-d*0.5)*np.abs(detm)**(-0.5)*np.exp(-0.5*self.quad(mu-nu,invm,mu-nu))
        return prob
    
    def log_zpq(self,mu,nu,sigma,tao):
        """
        Para el calculo de Cauchy-Schwarz 
        """
        d = float(len(mu))
        # isig = np.linalg.inv(sigma)
        # itau = np.linalg.inv(tao)
        # detm = np.linalg.det(isig+itau)
        # invm = np.linalg.inv(isig+itau)
        detm = np.linalg.det(sigma+tao)
        invm = np.linalg.inv(sigma+tao)
        prob = -d*0.5*np.log(2*np.pi) - 0.5*np.log(np.abs(detm)) - 0.5*self.quad(mu-nu,invm,mu-nu)
        return prob
        
    def js_div(self,mu,sigma,nu,tau):
        """
        Computes the Jensen-Shannon divergence measure.
        This measure is symmetric
        """
        m_mu = 0.5*(mu+nu)
        m_sigma = 0.5**2*(sigma+tau)
        Hm = self.entropy_gauss(m_mu,m_sigma)-self.entropy_gauss(mu,sigma)-self.entropy_gauss(nu,tau)
        return Hm
        
    def divergence(self,div):
        dic = ["cs","kl","js"]
        if div not in dic:
            return "Enter a valid divergence"
        div_mat = np.zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(self.N):
                mu    =  self.model.mus[i]
                nu    =  self.model.mus[j]
                sigma =  self.model.covs[i]
                tao   =  self.model.covs[j]
                if div == "js":
                    div_mat[i][j] = self.js_div(mu,sigma,nu,tao)
                if div == "kl":
                    div_mat[i][j] = self.kl_div(mu,sigma,nu,tao)
                if div == "cs":
                    div_mat[i][j] = np.exp(-self.cs_div(mu,sigma,nu,tao))
        return div_mat
    
    def plot_div_mat(self,div="cs"):
        plt.figure("Divergence Matrix Fs",figsize=(4.5,3))
        plt.clf()
        if div == "cs":
            plt.title(r"Divergence Matrix $e^{-CS(P|Q)}$ FS")
            # plt.imshow(-np.log(div_mat))
            plt.imshow(self.div_mat)
        if div == "js":
            plt.title("Divergence Matrix JC(P|Q) FS")
            plt.imshow(self.div_mat)
        if div == "kl":
            plt.title("Divergence Matrix KL(P|Q) FS")
            plt.imshow(self.div_mat)
        plt.xticks(np.arange(self.N))
        plt.yticks(np.arange(self.N))
        plt.ylabel("P")
        plt.xlabel("Q")

        plt.colorbar()
        plt.tight_layout()
        plt.pause(0.02)
    
    
    def stream(self,its1=100,its2=100):
        """
        Simula un stream de datos, ni tiempo entre muestras, ni el tiempo de ejecucion de las rutinas de HMM.
        """
        self.t=1
        # Primer Entrenamiento
        if self.firstt == True:
            self.train_model(its1, its2)
        else:
            pass
        self.div_mat = self.divergence("cs")
        
        # Iniciar serie de BIC
        self.bict.append(self.model.bic/self.L)
        
        # Inicia parametros para el test secuencial de Page
        self.mean = self.model.bic/self.L
        self.Sn.append(self.mean)
        
        # Pinta el camino de Viterbi 
        self.viterbi = self.model.viterbi(self.x[:self.L],plot=False,indexes=True)
        self.viterbi_hist = self.model.viterbi(self.x[:self.L],plot=False)
        self.ps = self.viterbi[-1]
        self.rhot = self.model.rho[self.viterbi]
        
        # Saca los parametros base de buena salud
        self.base_health_index()
        
        # Primer computo de salud
        health0 = self.health_index_init(self.x[:self.L],self.mode)
        self.health = health0[1]
        self.healthk = health0[0]
        self.meanhi1 = np.mean(np.array(self.health))
        
        while(self.L+(self.t-1)*self.slide<len(self.x)):
            if np.mod(self.t,25) == 0:
                self.plot_measures()
            # LLegada de nueva señal
            if self.L+self.t*self.slide<=self.Lmax:
                xt = self.x[:self.L+self.t*self.slide]
            if self.L+self.t*self.slide>self.Lmax:
                xt = self.x[self.L+self.t*self.slide-self.Lmax:self.L+self.t*self.slide]
            if self.L+self.t*self.slide>len(self.x):
                xt = self.x[self.t*self.slide-self.Lmax:]  
            leng = float(len(xt))
            print("Longitud actual: " + str(leng))
            
            # Hacer test de Page
            cbic = self.model.log_likelihood(xt,pena=True,ps=self.ps)/leng
            nlab = self.page_hinkley(self.it+1,cbic)
            if np.mod(self.t,50) == 0:
                self.plot_page_hinkley()
            
            # Determinación el numero de anomalias en ultima ventana
            self.label = np.roll(self.label,-1)
            self.label[-1] = nlab
            pros1 = self.test_bernoulli(self.label)
            if pros1 == True:
                self.train_model_run(xt,its1,its2,ps=self.ps)
                self.plot_div_mat()
                
            # Revision de BIC
            cbic = self.model.log_likelihood(xt,pena=True,ps=self.ps)/leng
            health1         = self.health_index_new(xt,self.mode)
            print("Current BIC:" + str(cbic))
            self.bict.append(cbic)
            
            self.health     = np.concatenate([self.health,health1[1]],axis=0)
            self.health     = (self.movmean(self.health.T[0],20))[np.newaxis].T
            
            self.healthk    = np.concatenate([self.healthk,health1[0]],axis=0)
            for k in range(len(self.healthk[0])):
                self.healthk[:,k]     =  ((self.movmean(self.healthk[:,k],20))[np.newaxis].T)[:,0]

            # Actualizo ps
            self.viterbi = self.model.viterbi(xt,plot=False,indexes=True,ps=self.ps)
            vit_h = self.model.viterbi(xt,plot=False,ps=self.ps)[-self.slide:]
            self.viterbi_hist =np.concatenate([self.viterbi_hist,vit_h])
            
            # Actualizar plots de feature saliency
            if self.asym == True:
                if self.L+self.t*self.slide<len(self.x):
                    self.rhot = np.concatenate([self.rhot,self.model.rho[self.viterbi[-self.slide:]]],axis=0)
                else:
                    diff = (self.L+(self.t-1)*self.slide) -len(self.x)
                    self.rhot = np.concatenate([self.rhot,self.model.rho[self.viterbi[-diff:]]],axis=0)
            else:
                if self.L+self.t*self.slide<len(self.x):
                     self.rhot = np.concatenate([self.rhot,np.tile(self.model.rho,self.slide)],axis=0)
                else:
                    diff = (self.L+(self.t-1)*self.slide) -len(self.x)
                    self.rhot = np.concatenate([self.rhot,np.tile(self.model.rho,diff)],axis=0)
            if np.mod(self.t,50) ==0:
                self.plot_fs_evolution()
            
            # Actualizar ps
            if self.L+self.t*self.slide<=self.Lmax:
                self.ps = self.viterbi[0]
            if self.L+self.t*self.slide>self.Lmax:
                self.ps = self.viterbi[-self.Lmax+self.model.P]
            if self.L+self.t*self.slide>len(self.x):
                self.ps = self.viterbi[0]
                
            # Computo de RUL
            if self.RUL_compute == True:
                self.predict_RUL()
                if np.mod(self.t,50) == 0:
                    self.plot_RUL()
            if np.mod(self.t,50) == 0:
                self.plot_hidden_states()
            
            # Actualiza plot del BIC
            if np.mod(self.t,50) == 0:
                self.refresh_bic()
            print("current t:"+ str(self.t))
            self.t = self.t+1     
            self.it = self.it+1
        return [self.bict,self.Sn,self.health, self.healthk,self.RUL]


class poly:
    '''
    Esta clase determina un polinomio por sus coeficientes
    si poly = [1,2,3,4,5]
    entonces f(x) = x**4+2x**3+3x**2+4x+5
    '''
    def __init__(self):
        return None
    
    def pol_plot(self,poli,t,T,dt =0.1,nombre = "Polynomial plot",ylim=-5):
        """
        plots a polynomial in[t0-T,t0+T)
        it assumes that:
            f(x) = poli[-1]+poli[-2]*x0+poli[-3]*x0**2+...
        """
        N = len(poli)-1
        times = np.arange(t-T,t+T,dt)
        feval = poli[-1]*np.ones(len(times))
        for i in range(0,len(poli)-1):
            feval += poli[i]*times**(N-i)
        plt.figure(nombre)
        # plt.clf()
        plt.title(nombre)
        plt.plot(times,feval)
        # plt.ylim(bottom=ylim,top=0)
        plt.grid("on")
        plt.pause(0.02)
            
        
        
    def pol_eval(self,poli,x0):
        """
        Evaluates the polynomial at a given x0
        it assumes that:
            f(x) = poli[-1]+poli[-2]*x0+poli[-3]*x0**2+...
        """
        eva= 0.0
        deg = len(poli)-1
        for i in range(len(poli)):
            eva += poli[i]*x0**(deg-i)
        return eva
    
    def pol_dif(self,poli):
        """
        Derives a polynomial
        it assumes that:
            f(x) = poli[-1]+poli[-2]*x0+poli[-3]*x0**2+...
        """
        poli = poli[::-1]
        npol = np.zeros(len(poli)-1)+0j
        for i in range(1,len(poli)):
            npol[i-1] = (i+0j)*poli[i]
        return npol[::-1]
    
    def pol_int(self,poli,x0=0,y0=0):
        """
        Integrates the polynomial
        it assumes that:
            f(x) = poli[-1]+x*poli[-2]+x**2+poli[-3]+...
        """
        poli = np.array(poli)[::-1]
        npol = np.zeros(len(poli)+1)+0j
        for i in range(1,len(poli)+1):
            npol[i] = poli[i-1]/(i+0j)
        npol = npol[::-1]
        a0 = self.pol_eval(npol, x0)
        npol[-1] = y0-a0
        return npol
        
            
    def newton_poli(self,poli,x0,its=400,err=1e-16):
        """
        Does the Newthon-rhapson algorithm for a polynomial, finds a root if possible
        """
        x1 = x0+0.5j
        dpoli = self.pol_dif(poli)
        for i in range(its):
            x2 = x1 -self.pol_eval(poli,x1)/self.pol_eval(dpoli,x1)
            if(np.abs(x2-x1)< err):
                break
            x1=x2
        return x1
    
    def pol_division(self,divisor,dividend):
        """
        divides dividend by divisor
        this assumes that the polynomials have unitary leading coefficient 
        """
        if len(dividend)< len(divisor):
            return "First argument must have higher order than the second one"
        else:
            out = np.copy(dividend)
            for i in range(len(dividend)-len(divisor)+1):
                coef = out[i]
                if coef != 0: 
                    for j in range(1, len(divisor)):  
                        out[i + j] += -divisor[j] * coef
            separator =1-len(divisor)
            q = out[:separator]
            r = out[separator:]
            return [q,r]
                
    def find_zeros_AR(self,B,p,i,m): #TODO existen metodos mas rapidos y que poseen mayor certeza, mirar matriz de compania.
        """
        Determines the zeros of the characteristic polynomial of the AR process
        of variable m and hidden state i. Uses newton-Rhapson and long division.
        """
        if p[i][m]>0:
            poli = np.concatenate([[1],-1*B[i][m][-p[i][m]:]])
            deg = len(poli)-1
            poli = poli/poli[-1]
            roots = []
            q = np.array(poli)+0.5j
            for j in range(deg):
                rooti = self.newton_poli(q,q[-1])
                divisor = [1.,-rooti]
                roots.append(rooti)
                q,r = self.pol_division(divisor,q)
            return roots
        else:
            return "No AR values avaliable for hidden state " + str(i) + " and variable " + str(m)
    
    def plot_zeros(self,roots,i,m,K,N,dictionary):
        circle = np.exp(-1j*2.0*np.pi*np.arange(360)/360.)
        roots = np.array(roots)
        W = [x.real for x in circle]
        Z = [x.imag for x in circle]
        plt.subplot(N,K,m+i*K+1)
        val = np.max(np.abs(roots))
        if  math.isnan(val) or np.abs(val) == np.inf :
            val=100
        plt.title("i="+str(i)+" ,k="+ str(m),fontsize=8)
        plt.xlim([-2-val,2+val])
        plt.ylim([-2-val,2+val])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        X = [x.real for x in roots]
        Y = [x.imag for x in roots]
        if np.max(np.abs(roots)) >1:  
            plt.plot(W,Z, color='green')
        else:
            plt.plot(W,Z, color='blue')
        plt.scatter(X,Y, color='red')
        plt.grid("on")
        plt.tight_layout()
        plt.show()
        
    def all_roots(self,poli):
        """
        """
        roots = []
        deg = len(poli)-1
        q = np.array(poli)
        for j in range(deg):
            rooti = self.newton_poli(q,q[-1])
            divisor = [1.,-rooti]
            roots.append(rooti)
            q,r = self.pol_division(divisor,q)
        return [roots,r]
        
        
    def plot_AR(self,B,p,N,K,dictionary):
        plt.figure("Roots of AR polynomial for hidden states ",figsize=(4.5,3))
        plt.clf()
        plt.subplots_adjust(hspace = 0.6)
        plt.subplots_adjust(wspace = 0.6)
        for i in range(N):
            for k in range(K):
                rootsim = self.find_zeros_AR(B,p,i,k)
                if isinstance(rootsim,str) != True:
                    self.plot_zeros(rootsim,i,k,K,N,dictionary)
                else:
                    circle = np.exp(-1j*2.0*np.pi*np.arange(360)/360.)
                    W = [x.real for x in circle]
                    Z = [x.imag for x in circle]
                    plt.subplot(N,K,k+i*K+1)
                    plt.title("i="+str(i)+", k="+ str(k),fontsize=8)
                    plt.xlim([-2,2])
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                    plt.ylim([-2,2])
                    plt.plot(W,Z, color='blue')
                    plt.grid("on")
                    plt.tight_layout()
                    plt.show()
        plt.pause(0.02)
        
        
class linear_reg: 
    def __init__(self,X,Y,constant = True):
        """
        Matriz para aprender las medias es:
            beta = (X^TX)**-1X^Ty
        """
        self.constant = constant
        self.sigma = None
        self.X =X
        self.Y =Y
        self.R2bar = None
        self.Ypred = None
        # self.K =K
        self.K = len(np.array(X[0]))
        self.Ymean = np.mean(Y)
        self.Tf = len(Y)
        if constant == True:
            cons = np.ones(self.Tf)
            cons = cons[np.newaxis].T
            self.X = np.concatenate([cons,self.X],axis=1)
        self.beta = []
    
    def fit(self): 
        """
        Fit the data minimizing the least squares
        """
        p = np.linalg.inv(np.dot(self.X.T,self.X))
        P = np.dot(p,self.X.T)
        self.beta = np.dot(P,self.Y)
        
        self.Ypred = np.dot(self.X,self.beta)
        self.R2bar = self.r2_bar()
        self.sigma_est()
    
    def r2(self):
        """
        The score of a linear model
        """
        if len(self.beta)>0:
            
            R2 = 1-np.sum((self.Ypred-self.Y)**2)/np.sum((self.Ymean-self.Y)**2)
        return R2
    
    def r2_bar(self):
        """
        Computes the adjusted r2:
            r2_bar = 1-(1-r2)*(Tf-K)/(n-p-1)
        """
        K = self.K
        r2_bar = 1-(1-self.r2())*(self.Tf-K)/(self.Tf-K-1.)
        return r2_bar
            
    def pred(self,x): 
        """
        Predict values from the regression
        """
        if self.constant == True:
            t = len(x)
            cons = np.ones(t)
            cons = cons[np.newaxis].T
            x = np.concatenate([cons,x],axis=1)
        return np.sum(self.beta*x,axis=1)
    
    def sigma_est(self):
        """
        Estimates the standard deviation of the error term
        """
        self.sigma = np.sqrt(np.sum((self.Ypred-self.Y)**2)/(self.Tf-2))
        
        
class atrium: 
    
    def __init__(self):
        return None
    def hann_window(self,N):
        """
        Crea una ventana de Hann de tamaño N
        """
        x = np.arange(N)
        return np.cos(np.pi*(x-N/2)/N)**2
    
    def kurtosis(self,x,N):
        '''
        Calcula la Kurtosis de una señal x
        '''
        x = np.array(x)
        x = (x-np.sum(x)/N)
        k1 = np.sum(np.abs(x)**4)/float(N) 
        k2 = np.sum(np.abs(x)**2)/float(N)
        return k1/(k2**2)-2.0   
    
    def build_FIR(self,HB1,HB2,N,size):
        """
        N es el orden del filro
        size es el orden de la señal a filtrar
        """
        w = np.zeros(N)
        for i in range(N):
            if i != N//2:
                w[i] = (0.5-0.5*np.cos(2*np.pi*i/N))*(np.sin(2*np.pi*HB2*(i-N/2))- np.sin(2*np.pi*HB1*(i-int(N/2))))/(np.pi*(i-int(N/2)))
            else:
                w[i] = (0.5-0.5*np.cos(2*np.pi*i/N))*2*(HB2-HB1)
        filter_w = np.zeros(size)
        filter_w[:int(N/2)] = (w[:int(N/2)])[::-1]
        filter_w[-int(N/2):] = (w[int(N/2):])[::-1]
        filter_w = np.abs(np.fft.fft(filter_w))
        return filter_w
    
    def hi(self,t,beta,fr):
        """
        beta es el Parametro de decaimiento
        fr es la frecuencia de resonancia
        """
        ind = np.where(t<=0)[0]
        s = np.exp(-beta*t)*np.sin(2.*np.pi*fr*t)
        s[ind] = 0
        return s
    
    def init_filtros(self,HB,num_ventanas,order,m_size_adc):
        """ 
        Genera los filtros pasabajas para el EASK
        filtros es una matriz
        HB es la frecuencia de corte del filtro base, es un numero entre 0 y 1
        num_ventanas es el numero de ventanas
        center es la longitud del filtro
        """
        filtros = []
        for i in range(num_ventanas):
            filtros.append(self.build_FIR(0,HB*(i+1),order,m_size_adc))
        return np.array(filtros)
    
    def init_senales(self,x,filtros,num_ventanas):
        """
        Filtra las señales por los filtros pasabajas
        """
        x_filter = []
        x_hat = np.fft.fft(x)
        for i in range(num_ventanas):
            x_n = np.real(np.fft.ifft(x_hat*filtros[i]))
            x_filter.append(x_n)
        return np.array(x_filter)
    
    def EASK(self,data_x,senales_filtradas,m_size_adc,num_ventanas,m_window,k_w,plot =True):
        """
        Selecciona el acho de banda con mayor kurtosis o mixtura kurtosis-energia
        se retorna la señal filtrada con tal ancho de banda.
        """
        energias = np.zeros([num_ventanas,num_ventanas])        
        resultados = np.zeros([num_ventanas,num_ventanas])  
        total_energy = np.sum(data_x**2)
        for a in range(num_ventanas):
            for b in range(a,num_ventanas):
                wind_energy =0
                temporal_x = np.zeros(m_size_adc)
                if (a==0):
                    temporal_x = senales_filtradas[b]
                else:
                    temporal_x = senales_filtradas[b]-senales_filtradas[a-1]
                wind_energy = np.sum(temporal_x**2)
                energias[a][b] = wind_energy/total_energy/ (b-a+1) 
                resultados[a][b] = self.kurtosis(temporal_x,m_size_adc) 
        mixture = (1-k_w)*energias/np.max(energias)+k_w*resultados/np.max(resultados)
        maxi = np.where(np.max(mixture)== mixture)
        fila = maxi[0][0]
        columna = maxi[1][0]
        temporal_x = np.zeros(m_size_adc)
        if(fila==0):
            temporal_x= senales_filtradas[columna]
        else:
            temporal_x= senales_filtradas[columna]-senales_filtradas[fila-1]
            
        if plot ==True:
            tic = np.arange(num_ventanas)[::3]*m_window
            plt.figure("Resultados Kurtosis")
            plt.clf()
            plt.title("Kurtosis matrix")
            plt.xlabel("Frecuencia final [Hz]")
            plt.ylabel("Frecuencia inicial [Hz]")
            tic = np.arange(num_ventanas)[::3]*m_window
            plt.yticks(np.arange(num_ventanas)[::3],tic)
            plt.xticks(np.arange(num_ventanas)[::3],tic,rotation=90)
            plt.imshow(resultados, cmap = "jet")
            plt.colorbar()
            plt.tight_layout()
            
            plt.figure("Resultados Energia")
            plt.clf()
            plt.title("Energy matrix")
            plt.xlabel("Frecuencia final [Hz]")
            plt.ylabel("Frecuencia inicial [Hz]")
            plt.yticks(np.arange(num_ventanas)[::3],tic)
            plt.xticks(np.arange(num_ventanas)[::3],tic,rotation=90)
            plt.imshow(energias, cmap = "jet")
            plt.colorbar()
            plt.tight_layout()
            
            plt.figure("Resultados Mixtura")
            plt.clf()
            plt.title("Mixture matrix")
            plt.xlabel("Frecuencia final [Hz]")
            plt.ylabel("Frecuencia inicial [Hz]")
            plt.yticks(np.arange(num_ventanas)[::3],tic)
            plt.xticks(np.arange(num_ventanas)[::3],tic,rotation=90)
            plt.imshow(mixture, cmap = "jet")
            plt.colorbar()
            plt.tight_layout()
        return  temporal_x
    
    def freq_search(self,fft,target,frecuencia,df,tolerance,mfft_size):
        """
        Busqueda de frecuencias
        fft es el modulo de la transformada de fourier a buscar
        target es un vector de longitud dos donde se guardan los resultados
        frecuencia es la frecuencia de muestreo
        df es la resolucion en espectro
        tolerance es la tolerancia de la banda de busqueda en el espectro
        mfft_size es el tamaño de fft
        """
        center = np.fix(frecuencia/df)
        tolerance = np.fix(tolerance/df)
        maxi = -1.
        real_freq =-1.
        if (center+ tolerance <mfft_size/2.):
            for i in range(int(center-tolerance),int(center+tolerance)):
                if(maxi <= fft[i]):
                    real_freq = i*df
                    maxi = fft[i]
        target[0] = real_freq
        target[1] = maxi
        return None
    
    def atrium(self,x,m_size_adc,m_order, m_tolerance, m_freqm, m_window, m_num_har, m_fr,dt, k_w, BPFO, BPFI, FTF, BSF,plot=True):
        """
        m_size_adc:  Tamaño original de la señal, debe ser potencia de dos
        m_order:     Ordén/ longitud  del filtro FIR
        m_tolerance: Tolerancia para buscar en armonicos
        m_freqm:     Frecuencia de muestreo
        m_window:    Tamaño de filtro inicial del EASK en frecuencia(Hz)
        m_num_har:   Número de armónicos a mirar
        m_fr:        Frecuencia de giro
        dt:          Delta de tiempo
        k_:          Peso de la kurtosis en la mixtura
        BPFO:        Frecuencia de pista externa
        BPFI:        Frecuencia de pista interna
        BSF:         Frecuencia de bola
        FTF:         Frecuencia de jaula
        """
        df = float(m_freqm)/float(m_size_adc) 
        num_ventanas = int(np.round(0.5*(m_freqm/m_window)-1)) #Numero de ventanas o de filtros base pasabajas
        HB = m_window/m_freqm # Ancho de banda relativo
        filtros = self.init_filtros(HB, num_ventanas,m_order,m_size_adc)
        x_mod = x*self.hann_window(m_size_adc)
        x_filtered =self.init_senales(x_mod,filtros,num_ventanas)
        temporal_x = self.EASK(x_mod,x_filtered,m_size_adc,num_ventanas,m_window,k_w,plot=plot)
        envelope = np.abs(hilbert(temporal_x))
        spectrum = np.abs(np.fft.fft(envelope))*2/m_size_adc
        spectrum[0] = 0   
        target = np.zeros([5*m_num_har,2])
        for j in range(m_num_har):
            self.freq_search(spectrum,target[0+5*j],(j+1)*m_fr,df,m_tolerance,m_size_adc)
            self.freq_search(spectrum,target[1+5*j],(j+1)*BPFO,df,m_tolerance,m_size_adc)
            self.freq_search(spectrum,target[2+5*j],(j+1)*BPFI,df,m_tolerance,m_size_adc)
            self.freq_search(spectrum,target[3+5*j],(j+1)*FTF,df,m_tolerance,m_size_adc)
            self.freq_search(spectrum,target[4+5*j],(j+1)*BSF,df,m_tolerance,m_size_adc)
        freq = np.arange(m_size_adc)*m_freqm/float(m_size_adc)        
        #Plots
        if plot==True:
            
            plt.figure("Raw spectrum")
            plt.clf()
            plt.title("Raw spectrum")
            plt.plot(freq,np.abs(np.fft.fft(x))*2/float(len(temporal_x)))
            plt.xlabel("Frecuencia Hz")
            plt.ylabel("Magnitud")
            plt.grid("on")    
            plt.tight_layout()
            
            
            plt.figure("Resonance band")
            plt.clf()
            plt.title("Resonance band")
            plt.plot(freq,np.abs(np.fft.fft(temporal_x))*2/float(len(temporal_x)))
            plt.xlabel("Frecuencia Hz")
            plt.ylabel("Magnitud")
            plt.grid("on")
            plt.tight_layout()
        
            plt.figure("Envelope spectrum")
            plt.clf()
            plt.title("Envelope spectrum")
            plt.plot(freq,spectrum)
            plt.xlim([0,np.max([m_fr,BPFO,BPFI,FTF,BSF])+10])
            plt.ylim([0,np.max(spectrum)])
            plt.axvline(x=BPFO,c ="r",linestyle="--")
            plt.axvline(x=BPFI, c = "g",linestyle="--")
            plt.axvline(x=FTF, c = "b",linestyle="--")
            plt.axvline(x=BSF, c = "yellow",linestyle="--")
            plt.xlabel("Frecuencia Hz")
            plt.ylabel("Magnitud")
            plt.grid("on")
            plt.tight_layout()
        return [target,spectrum,freq]
        


    
