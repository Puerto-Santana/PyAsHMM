# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:37:11 2021

@author: epuerto
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import colorsys
# from AR_ASLG_HMM import AR_ASLG_HMM as hmm
from FS_AR_ASLG_HMM import FS_AR_ASLG_HMM as hmm
# from FS_AR_ASLG_HMM import FS_AR_ASLG_HMM as hmm
class fss_stream:
    def __init__(self, O, K, N, L, dt, slide, pp=0.1, alpha=3,gamma2=0.05,zthr=0.9,label=None,act_inde=True,max_leng=4000 ):
        """
        """
        self.max_l = max_leng
        self.relevancy_times = None
        self.act_inde =act_inde                                # Boolean que determina si se actualiza el rudio   
        self.dt = dt
        self.O      = O                                        # Observaciones para el data-steam
        self.K      = K                                        # Numero de variables
        self.N      = N                                        # Numero de estados ocultos
        self.L      = L                                        # Tamaño de la ventana 
        self.slide  = slide                                    # Numero de datos que se desplaza 
        self.pp     = pp                                       # Porcentaje de anomalias para el test de page
        self.gamma  = L*np.log(alpha)                          # Threshold del test secuencial de Page
        self.T      = len(O)                                   # Longitud de los datos
        self.gamma2 = gamma2                                   # Confiabilidad cotas de Chernoff
        self.Ns = self.min_sample(gamma2,pp,1e-2)              # ventana de datos para cotas de chernoff   
        self.label_dif_reduced_local   = np.zeros([self.Ns,K]) # Contador de anomalias en el modelo reducido local
        self.label_reduced_full        = np.zeros(self.Ns)     # Contador de anomalias en el modelo reducido full
        self.label_neglect             = np.zeros([self.Ns,K]) # Contador de anomalias en el ruido local de mod reducido
        self.mean_dif_reduced_local    = np.zeros(self.K)      # Log-verosimilitud media en el modelo reducido local
        self.mean_reduced_full       = 0                     # Log-verosimilitud media en el modelo reducido full
        self.mean_neglect              = np.zeros(self.K)      # Log-verosimilitud media en el ruido global 
        self.Sn_dif_reduced_local      = []
        self.Sn_reduced_full         = []
        self.Sn_neglect                = []
        self.sel_var                   = np.zeros(self.K)      # Vector de flags que indica que variables son relevantes
        self.g_model                   = None                  # Modelo global es fshmm
        self.r_model                   = None                  # Modelo reducido es hmm_naive
        self.z_thr                     = zthr                  # Threhsold para determinar que variables pueden ser relevantes
        
        self.hll_dif_reduced_local     = []                    # Guarda el historico de la verosimilitud del modelo reducid
        self.hll_reduced_full          = []                    # Guarda el historico de la verosimilitud del modelo reducido global
        self.hll_neglect               = []                    # Guarda el historico de la verosimilitud de las variables rechazadas
        
        self.h_sel_var                 = []                    # Guarda el historico de variables seleccionadas
        self.h_weig                    = []                    # Guarda el historico de pesos 
        
        self.label_global_full         = np.zeros(self.Ns)
        self.mean_global_full          = 0
        self.Sn_global_full            = []
        self.hll_global_full           = []
        self.h_ph                      = [0]
        self.label                     = label
        self.nd_t                      = []                    # Concept drifts
        
    def color(self,k):
        HSV_tuples = [(x*1.0/k, 0.5, 0.75) for x in range(k)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = colorsys.hsv_to_rgb(*rgb)
            hex_out.append(rgb)
        return hex_out

    
    def train_model_t0(self,o):
        """
        """
        self.g_model = fshmm(o,np.array([self.L]),self.K,self.N,act_inde=self.act_inde)
        self.g_model.EM()
        self.act_rel()
        ind = np.argwhere(self.sel_var ==1).T[0]
        self.r_model = hmm((o.T[ind]).T, np.array([self.L]),len(ind),self.N,P=0)
        self.r_model.EM()
        self.h_sel_var.append(np.copy(self.sel_var))
        
    def update_model(self,changes,o):
        """
        """
        c_rl = changes[0] # variables a retirar del modelo reducido
        c_rg = changes[1] # modificar el modelo reducido
        c_ng = changes[2] # variables a agregar al modelo reducido
        
        if np.sum(c_rl)+np.sum(c_ng) != 0  :
            for i in range(self.K):
                if c_rl[i] == 1:
                    self.sel_var[i] = 0
                if c_ng[i] == 1:
                    self.sel_var[i] = 1
            self.g_model = fshmm(o,np.array([self.L]),self.K,self.N,act_inde=self.act_inde)
            self.g_model.EM()
            self.act_rel()
            ind = np.argwhere(self.sel_var ==1).T[0]
            self.r_model = hmm((o.T[ind]).T, np.array([self.L]),len(ind),self.N,P=0)
            self.r_model.EM()
            self.reset_novel(o)
            self.it_r = 1
            self.it_g = 1
        if c_rg ==1  and np.sum(c_rl)+np.sum(c_ng) == 0  :
            ind = np.argwhere(self.sel_var ==1).T[0]
            self.r_model = hmm((o.T[ind]).T, np.array([self.L]),len(ind),self.N,P=0)
            self.r_model.EM()
            self.label_reduced_full = np.zeros(self.Ns)
            ind = np.argwhere(self.sel_var ==1).T[0]
            self.Sn_reduced_full = [self.r_model.log_likelihood((o.T[ind]).T)]
            self.mean_reduced_full = self.Sn_reduced_full[0]
            self.it_r = 1
            
            
        [ll_dif_reduced_local , ll_reduced_full, ll_neglect ] = self.compute_all_ll(o)
        self.hll_reduced_full.append(ll_reduced_full)
        self.hll_dif_reduced_local.append(ll_dif_reduced_local)
        self.hll_neglect.append(ll_neglect)
        print("Current used variables to save: " + str(self.sel_var))
        self.h_sel_var.append(np.copy(self.sel_var))
            
        
            
    def reset_novel(self,o):
        """
        """
        self.label_dif_reduced_local   = np.zeros([self.Ns,self.K])   # Contador de anomalias en el modelo reducido local
        self.label_reduced_full      = np.zeros(self.Ns)           # Contador de anomalias en el modelo reducido full
        self.label_neglect             = np.zeros([self.Ns,self.K])       # Contador de anomalias en el ruido local de mod reducido
       
        [ll_dif_reduced_local , ll_reduced_full, ll_neglect ] = self.compute_all_ll(o)
        
        self.mean_dif_reduced_local = ll_reduced_full
        self.mean_reduced_full    = ll_reduced_full     
        self.mean_neglect           = ll_neglect  
        
        self.Sn_reduced_full     = [ll_reduced_full]
        self.Sn_dif_reduced_local   = [ll_dif_reduced_local]
        self.Sn_neglect             = [ll_neglect]
        
    def compute_all_ll(self,o):
        """
        """
        ll_dif_reduced_local    = self.compute_dif_reduced_local_likelihood(o)  
        ll_reduced_full       = self.compute_reduced_full_likelihood(o)           
        ll_neglect              = self.compute_neglect_likelihood(o)
        return [-2*ll_dif_reduced_local , -2*ll_reduced_full, -2*ll_neglect ]
        
        
    def act_rel(self):
        """
        """
        z_re = self.g_model.rho
        for k in range(self.K):
            if z_re[k] >=self.z_thr:
                self.sel_var[k] =int(1)
            else:
                self.sel_var[k] = int(0)
        self.sel_var.astype(int)
                
    def compute_neglect_likelihood(self,O):
        """
        """
        fb_g = self.g_model.forBack[0]
        ll = np.ones(self.K)
        for i,x in enumerate(self.sel_var):
            if x==0:
                ll[i] = fb_g.log_likelihood_ind_k(O,i,self.g_model.mu,self.g_model.sigma,self.g_model.epsi,self.g_model.tau)
        return ll
    
    def compute_dif_reduced_local_likelihood(self,O):
        """
        """
        fb_r = self.r_model.forBack[0]
        fb_g = self.g_model.forBack[0]
        ll = np.ones(self.K)
        ind = np.argwhere(self.sel_var ==1).T[0]
        for i,x in enumerate(ind):
            ll[x] = fb_r.log_likelihood_k(O,i,self.r_model.A,self.r_model.pi,
                                          self.r_model.B,self.r_model.G,self.r_model.N,
                                          self.r_model.p,self.r_model.P,self.r_model.K,
                                          self.r_model.sigma)
            ll[x] -= fb_g.log_likelihood_ind_k(O,x,self.g_model.mu,self.g_model.sigma,self.g_model.epsi,self.g_model.tau)
        return ll
    
    def compute_reduced_full_likelihood(self,O):
        """
        """
        ind = np.argwhere(self.sel_var ==1).T[0]
        return self.r_model.log_likelihood((O.T[ind]).T)
    
    def compute_global_full_likelihood(self,O):
        ll = -2*self.g_model.log_likelihood(O)
        return ll/float(len(O))
        
    def page_neglect(self,t,ot):
        """
        retorna un vector indicando que variables  omitidas observan anomalias
        con respecto a su comportamiento de ruido
        """
        nmu = ot/t+(t-1)/t*self.mean_neglect 
        S = nmu
        self.Sn_neglect.append(S)
        self.mean_neglect = nmu
        Ph = S-np.min(np.array(self.Sn_neglect),axis=0)
        nlab = np.zeros(self.K)
        for k,x in enumerate(self.sel_var):
            if x==0 and Ph[k] > self.gamma:
                nlab[k] = 1
        return nlab
    
    def page_reduced_full(self,t,ot):
        """
        """
        nmu = ot/t+(t-1)/t*self.mean_reduced_full
        S = nmu
        self.Sn_reduced_full.append(S)
        self.mean_reduced_full = nmu
        Ph = S-np.min(np.array(self.Sn_reduced_full))
        nlab = 0
        if Ph > self.gamma:
            nlab  = 1
        return nlab
    
    def page_global_full(self,t,ot):
        """
        """
        nmu = ot/t+(t-1)/t*self.mean_global_full
        S = nmu
        self.Sn_global_full.append(S)
        self.mean_global_full = nmu
        Ph = S-np.min(np.array(self.Sn_global_full))
        self.h_ph.append(Ph)
        # print("test: " +str(Ph) + ", threshold: " +str(self.gamma/float(self.L)))
        nlab = 0
        if Ph > self.gamma/float(self.L):
            nlab  = 1
        return nlab
    
    def page_dif_reduced_local(self,t,ot):
        """
        """
        nmu = ot/t+(t-1)/t*self.mean_dif_reduced_local
        S = nmu
        self.Sn_dif_reduced_local.append(S)
        self.mean_reduced_local = nmu
        Ph = S-np.min(np.array(self.Sn_dif_reduced_local),axis=0)
        nlab = np.zeros(self.K)
        for k,x in enumerate(self.sel_var):
            if x==1 and Ph[k] > self.gamma:
                nlab[k] = 1
        return nlab

    def min_sample(self,delta,p,epsilon): 
        n= -np.log(1-delta)/self.bernoulli_kullback_leibler(p-epsilon, p)
        print("Size of outliers window: "+ str(n))
        return int(np.ceil(n))
    
    def bernoulli_kullback_leibler(self,p,q): 
        """
        Determina la divergencia de Kullback-Leibler de dos distribuciones Bernoulli 
        D(p|q) = qlog(q/p)+(1-q)log((1-q)/(1-p))
        """
        return q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))

    def test_bernoulli(self):
        c_rl = np.zeros(self.K)
        c_ng = np.zeros(self.K)
        c_rg = 0
        p_rl = np.mean(self.label_dif_reduced_local,axis=0)
        p_ng = np.mean(self.label_neglect ,axis=0)
        p_rg = np.mean(self.label_reduced_full)
        for k in range(self.K):
            if p_rl[k] >self.pp:
                c_rl[k] =1
            if p_ng[k] > self.pp:
                c_ng[k] = 1
        if p_rg > self.pp:
            c_rg =1
        return [c_rl,c_rg,c_ng]
    
    def test_bernoulli_global_full(self):
        c_gf = 0
        p_gf = np.mean(self.label_global_full)
        if p_gf> self.pp:
            c_gf =1
        return c_gf
            
        
    def stream(self):
        """
        """
        o = self.O[:self.L]
        self.train_model_t0(o)
        self.h_weig.append(self.g_model.rho)
        
        [ll_dif_reduced_local, ll_reduced_full, ll_neglect]= self.compute_all_ll(o)        
        
        self.mean_dif_reduced_local    = ll_dif_reduced_local  
        self.mean_reduced_full   = ll_reduced_full        
        self.mean_neglect          = ll_neglect  
        
        self.Sn_reduced_full.append(ll_reduced_full)
        self.Sn_dif_reduced_local.append(ll_dif_reduced_local)
        self.Sn_neglect.append(ll_neglect)
        
        self.hll_reduced_full.append(ll_reduced_full)
        self.hll_dif_reduced_local.append(ll_dif_reduced_local)
        self.hll_neglect.append(ll_neglect)
        t= 1
        self.it_r = 1
        self.it_g = 1
        while(self.L+(t-1)*self.slide<len(self.O)):
            if self.L+t*self.slide>len(self.O):
                xt = self.O[t*self.slide:]  
            else:
                xt = self.O[t*self.slide:self.L+t*self.slide] 
                
            nlab_rl = self.page_dif_reduced_local(self.it_r,  self.hll_dif_reduced_local[-1])
            nlab_rg = self.page_reduced_full(self.it_g, self.hll_reduced_full[-1])
            nlab_ng = self.page_neglect(self.it_g, self.hll_neglect[-1] )
            
            self.label_dif_reduced_local = np.roll(self.label_dif_reduced_local,-1,axis=0)
            self.label_reduced_full = np.roll(self.label_reduced_full,-1)
            self.label_neglect  = np.roll(self.label_neglect,-1,axis=0)
            
            
            self.label_dif_reduced_local[-1]  = nlab_rl
            self.label_reduced_full[-1] = nlab_rg
            self.label_neglect[-1]       = nlab_ng
            
            changes = self.test_bernoulli()
            self.update_model(changes,xt)
            self.h_weig.append(self.g_model.rho)
            print(t)
            t =t+1
        
        
    def stream_v2(self):
        o = self.O[:self.L]
        self.g_model = fshmm(o,np.array([self.L]),self.K,self.N,act_inde=self.act_inde)
        self.g_model.EM()
        self.h_weig.append(self.g_model.rho)
        
        self.mean_global_full = self.compute_global_full_likelihood(o)
        self.Sn_global_full.append(self.mean_global_full)
        self.hll_global_full.append(self.mean_global_full)
        t=1
        self.it_g = 1
        while(self.L+(t-1)*self.slide<len(self.O)):
            if self.L+t*self.slide>len(self.O):
                # xt = self.O[t*self.slide:] 
                if len(self.O)>self.max_l:
                    xt = self.O[-self.max_l:]
                else:
                    xt = self.O
            else:
                # xt = self.O[t*self.slide:self.L+t*self.slide] 
                if  self.L+t*self.slide > self.max_l:
                    xt = self.O[self.L+t*self.slide-self.max_l:self.L+t*self.slide]
                else:
                    xt = self.O[:self.L+t*self.slide]
            nlab_gf = self.page_global_full(self.it_g, self.hll_global_full[-1])
            self.label_global_full = np.roll(self.label_global_full,-1)
            self.label_global_full[-1] = nlab_gf
            changes = self.test_bernoulli_global_full()
            if changes ==1:
                self.g_model = fshmm(xt,np.array([len(xt)]),self.K,self.N,act_inde=self.act_inde)
                self.g_model.EM()
                self.act_rel()
                self.it_g = 1
                self.label_global_full  = np.zeros(self.Ns)
                self.mean_global_full   = self.compute_global_full_likelihood(xt)
                self.Sn_global_full     = [self.mean_global_full]
                self.nd_t.append((t*self.slide+self.L)*self.dt)
            ll_gf = self.compute_global_full_likelihood(xt)
            self.hll_global_full.append(ll_gf)
            self.h_weig.append(self.g_model.rho)
            if np.mod(t,50)==0:
                print("Current iteration is:" + str(t))
            t=t+1
        self.plot_v2()
        self.relevancy_time()
            
    def plot_v2(self,freq=False,arcelor=False):
        T = len(self.hll_global_full)
        time = np.arange(T)*self.dt*self.slide+self.L*self.dt
        
        plt.figure("Evolution of relevancy",figsize= (12,8))
        plt.clf()
        plt.ylabel(r"$\rho$")
        plt.xlabel("Time [h]")
        for i in range(len(self.nd_t)):
            plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
        for i in range(self.K):
            if self.label is None:
                plt.plot(time,(np.array(self.h_weig).T[i]).T, label ="feature "+str(i+1))
            else:
                plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = (1-i/self.K,0,i/self.K))
                # plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i])
        plt.grid("on")
        plt.legend()
        plt.tight_layout()
        
        plt.figure("Evolution of BIC per unit time",figsize= (12,8))
        plt.clf()
        plt.ylabel(r"BIC/u")
        plt.xlabel("Time [h]")
        for i in range(len(self.nd_t)):
            plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
        plt.plot(time,self.hll_global_full)
        plt.grid("on")
        plt.tight_layout()
        
        plt.figure("Evolution of page test",figsize= (4.5,3))
        plt.clf()
        plt.ylabel(r"Ph")
        plt.xlabel("Time")
        for i in range(len(self.nd_t)):
            plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
        plt.plot(time,self.h_ph)
        plt.plot(time,self.gamma/(self.L)*np.ones(T))
        plt.grid("on")
        plt.tight_layout()
        
        if freq == True:
            plt.figure("Evolution of relevancy of f1",figsize= (4.5,3))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(self.K):
                if  np.mod(i,4) ==0:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = (1-i/self.K,0,i/self.K))
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            plt.figure("Evolution of relevancy of f2",figsize= (4.5,3))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(self.K):
                if  np.mod(i,4) ==1:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = (1-i/self.K,0,i/self.K))
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            plt.figure("Evolution of relevancy of f3",figsize= (4.5,3))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(self.K):
                if  np.mod(i,4) ==2:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = (1-i/self.K,0,i/self.K))
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            plt.figure("Evolution of relevancy of f4",figsize= (4.5,3))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(self.K):
                if  np.mod(i,4) ==3:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = (1-i/self.K,0,i/self.K))
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            colors = ["green","red","yellow", "magenta", "gray"]
            i=3
            inds = np.argmax(np.array(self.h_weig).T[[0+i,4+i,8+i,12+i,16+i]].T,axis=1)
            plt.figure("Data")
            for i in range(len(inds)-1):
                plt.axvspan(time[i], time[i+1], alpha=0.1, color= colors[inds[i]])
            for k in range(5):
                plt.plot(time,np.zeros(T),color=colors[k],label=self.label[3+k*4])
            plt.ylim(bottom= 0.1)
            plt.legend(loc=4)
            
            inds2 = np.argmin(np.array(self.h_weig).T[[0+i,4+i,8+i,12+i,16+i]].T,axis=1)
            plt.figure("Data2")
            for i in range(len(inds)-1):
                plt.axvspan(time[i], time[i+1], alpha=0.1, color= colors[inds2[i]])
            for k in range(5):
                plt.plot(time,np.zeros(T),color=colors[k],label=self.label[3+k*4])
            plt.ylim(bottom= 0.1)
            plt.legend(loc=4)
            
        if arcelor==True:
            colores = self.color(15)
            plt.figure("Evolution of relevancy 1",figsize= (12,8))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(15):
                if self.label is None:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label ="feature "+str(i+1))
                else:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = colores[i])
                    # plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i])
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            colores2= self.color(16)
            plt.figure("Evolution of relevancy 2",figsize= (12,8))
            plt.clf()
            plt.ylabel(r"$\rho$")
            plt.xlabel("Time [h]")
            for i in range(len(self.nd_t)):
                plt.axvline(x=self.nd_t[i],color="black",ls="--",linewidth=1)
            for i in range(15,self.K):
                if self.label is None:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label ="feature "+str(i+1))
                else:
                    plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i],color = colores2[i-15])
                    # plt.plot(time,(np.array(self.h_weig).T[i]).T, label =self.label[i])
            plt.grid("on")
            plt.legend(loc=4)
            plt.tight_layout()
            
            
    def relevancy_time(self):
        """
        Retorna un vector con la cantidad de tiempo que se estuvo en el inframund
        """
        dictio = []
        for i in range(self.K):
            rel_time = np.argwhere(np.array(self.h_weig).T[i].T>self.z_thr).T[0]
            dictio.append([self.label[i], len(rel_time) ,len(rel_time)/len(np.array(self.h_weig).T[i].T)])
        self.relevancy_times = dictio 
            
                
          
            
        
        
            
                
        

