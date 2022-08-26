"""
Created on Thu Jul 19 14:07:03 2018
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0
"""
from sklearn.cluster import KMeans
import pickle
import sys
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
class LMSAR:    
    def __init__(self,O,lengths,K,N,p=None,P=None,A=None,pi=None, B=None,
                 sigma=None,m=1,p_cor=5,struc=True,lags=True,nums=None,dens=None):
        """
        Create an object AR-ASLG-HMM
        O: the observations, it has dimension (n_1+n_2,+...+n_M)xK, 
            where n_m are the number of observations of the first dataset
            and K is the number of variables
        lengths: is a numpy array with the lengts of observations 
        N: the number of hidden States
        p: is a matrix NxK with the number of time lags per state and variable 
            p[i][k] is the number of lags for variable k in state i
        P: is the maximun number of time lags allowed
        A: the transition Matrix type numpy
        pi: the initial distribution vector type numpy
        G: is a list of lists, where each list is a graph
        L: is a list with numpy arrays, where each array is a topological order
            of the corresponding index graph
        B: is a list with lists, where each list has the linear dependies coefficients
            of the corresponding index graph.
        sigma: is a list of vectors, where each vector has got the variance coefficients
            of the corresponding index graph
        m: number of possible initial relationships within context bayesian graphs.
        """
        self.kappa = None
        self.nu = None
        self.lags =lags
        self.mus = None
        self.covs = None
        self.O = O #Observacion
        self.LogLtrain = None #Verosimilitud de los datos de entrenamiento
        self.bic = None
        self.b = None # lista de numero de parametros 
        self.forBack = [] #lista de objetos forBack
        self.dictionary = None  # Orden de los estados segun severidad
        self.K = K #Numero de variables
        self.N = N #Numero de estados
        self.lengths = lengths #vector con longitudes de las observaciones       
        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            self.P = P
        if  A is None:
            self.A = np.ones([N,N])/N
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)/N
        else:
            self.pi= pi
        if B  is None:
            kmeans = KMeans(n_clusters=N).fit(O)
            clusts = kmeans.predict(O)
            B= []
            for i in range(N):
                indi = np.where(clusts==i)[0]
                Oi = O[indi]
                kmi = np.mean(Oi,axis=0)
                Bi = []
                for k in range(self.K):
                    # Bik = np.concatenate([[(i+1)*(np.max(O.T[k])-np.min(O.T[k]))/(self.N+1)+np.min(O.T[k])],np.zeros(self.P)])
                    Bik = np.concatenate([[kmi[k]],np.zeros(self.P)])
                    Bi.append(Bik)
                B.append(Bi) 
            self.B = B
        else :
            self.B=B
            
        if sigma is None:
            sigma =[]
            for k in range(self.K):
                sigmak = np.abs(np.min(O.T[k])-np.max(O.T[k]))*100.0
                sigma.append(np.array(sigmak))
            sigma = np.array(sigma)
            self.sigma = sigma
        else:
            self.sigma = sigma
        

    def correlogram(self,O,p=0):
        """
        """
        if p==0:
            p = int(len(O)/4.)
        n = len(O)
        y = O[::-1]
        mean = np.mean(y)
        var = [np.sum((y-mean)**2)/n]
        for k in range(1,p+1):
            ylag = np.roll(y,-k)
            num = np.sum((y[:-k]-mean)*(ylag[:-k]-mean))*(n-k)**(-1)
            var.append(num)
        rho = np.array(var)/var[0]
        return rho[1:]

    def pcorrelogram(self,O,p=0,title="",plot=True):
        """
        """
        if p==0:
            p = int(len(O)/4.)
        n = len(O)
        rho = self.correlogram(O,p)
        prho = [rho[0]]
        for k in range(1,p):
            a = rho[:k+1]
            tor = np.concatenate([rho[:k][::-1],[1.],rho[:k]])
            b = [tor[-k-1:]]
            for j in range(1,k+1):
                b.append(np.roll(tor,j)[-k-1:])
            b = np.array(b)
            coef = np.linalg.solve(b,a)
            prho.append(coef[k])
        prho = np.array(prho)
        lags = np.arange(1,p+1)
        ucof = np.ones(p)*1.96*(1./np.sqrt(n))
        icof = np.ones(p)*-1.96*(1./np.sqrt(n))
        if( plot == True):
            plt.figure("PCorrelogram"+title)
            plt.subplot(211)
            plt.title("Partial correlogram"+title)
            plt.bar(lags,prho, label = "Partial correlations")
            plt.plot(lags,ucof,linestyle=":",color="black",label ="Confidence bands")
            plt.plot(lags,icof,linestyle=":",color="black")
            plt.plot(lags,np.zeros(p),color = "black",linewidth =0.8)
            plt.xlabel("Time lags")
            plt.ylabel("Partial Correlation" + r"$\phi_{kk}$")
            plt.ylim([-1,1])
            plt.legend()
            plt.subplot(212)
            plt.title("Correlogram"+title)
            plt.bar(lags,rho, label = "Correlations")
            plt.plot(lags,ucof,linestyle=":",color="black",label ="Confidence bands")
            plt.plot(lags,icof,linestyle=":",color="black")
            plt.plot(lags,np.zeros(p),color = "black",linewidth =0.8)
            plt.xlabel("Time lags")
            plt.ylabel("Correlation "+ r"$\rho_k$" )
            plt.ylim([-1,1])
            plt.legend()
            plt.show
        prho = prho[np.newaxis:]
        return [rho,prho]
    
    def det_lags(self,p_cor=50,plot=False):
        """
        """
        n,d = self.O.shape
        indrho =  []
        energy = []
        thresh = 1.96*(1/np.sqrt(n))
        for i in range(d):
            Oi =self.O.T[i]
#            O = np.diff(self.O.T[i])
            [rhoi,prhoi] = self.pcorrelogram(Oi,p=p_cor,plot=False,title ="variable: "+ str(i))
            prhoi = np.abs(prhoi)
            energy.append(np.sum(prhoi**2))
        ind = np.argmax(np.array(energy))
        Om =self.O.T[ind]
        [rhoi,prhoi] = self.pcorrelogram(Om,p=p_cor,plot=plot,title = " variable: "+ str(ind))
        verdad = True
        j=0
        while verdad == True:
            if np.abs(prhoi[j])< thresh or j+1 == p_cor:
                indrho.append(j-1)
                verdad = False  
            j=j+1
        indrho = np.array(indrho)
        return int(np.max(indrho)+1)
         
#  States management
                        
    def mvn_param(self):
        """
        Obtains the parameters of each multi dimensional normal distribution 
            corresponding to each state
        """
        self.covs = self.sigma
        mus = []
        for i in range(self.N):
            musi = []
            for k in range(self.K):
                musik = self.B[i][k][0]/(1.-np.sum(self.B[i][k][1:]))
                musi.append(musik)
            mus.append(musi)
        self.mus = np.array(mus)


        
    def label_parameters(self,kappa,nu):
        """
        Change the parameters for labelling
        """
        if len(kappa)!= self.K:
            return "Wrong dimension"
        self.kappa = kappa
        self.nu = nu
        
    def states_order(self,maxg = False,absg = True):
        """
        Order the states depending on the energy of the mvn mean vector taking into account AR coefficients
        """
        mag = {}
        if self.kappa is not None or self.nu is not None:
            if maxg== False and absg == False:
                music = np.sum((self.mus-self.kappa)*self.nu,axis=1)
            if maxg== True and absg == False:
                music = np.max((self.mus-self.kappa)*self.nu,axis=1)
            if maxg== False and absg == True:
                music = np.max(np.abs(self.mus-self.kappa)*self.nu,axis=1)
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus-self.kappa)*self.nu,axis=1)
        else:
            if maxg== False and absg == False:
                music = np.sum((self.mus),axis=1) 
            if maxg== True and absg == False:
                music = np.max((self.mus),axis=1)     
            if maxg== False and absg == True:
                music = np.sum(np.abs(self.mus),axis=1) 
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus),axis=1) 
        index = np.argsort(music)
        j=0
        for i in index:
            mag[j] = music[i]
            j=j+1
        self.dictionary= [index,mag,]
        
    
    
#   HMM
    def log_likelihood(self,O,pena =False,xunit=False):
        """
        Calcula la log verosimilitud de una observacion dado unos paramatros
        A matriz de transicion
        pi distribucion inicial
        mu lista de mu_i, donde mu_i son las medias para el estado i
        sigma lista de sigam_i donde sigma_i son las desviaciones para estado i
        O es matriz de observacion
        retorn log verosimilitud de la observacion
        """
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K)
        fb.prob_BN_t(O,self.sigma,self.P)
        fb.forward_backward_As(self.A,self.sigma,self.pi,O,self.P)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if xunit == False:
            if pena == False:
                return log
            else:
                return -2*(log+self.pen(self.P,np.array([len(O)]))[0])
        else:
            if pena == False:
                return log/float(len(O))
            else:
                return (-2*(log+self.pen(self.P,np.array([len(O)]))[0]))/float(len(O))
     
        
        
    def viterbi_step(self,delta,A,fb,t):
        """
        Paso para determinar sucesion de estados mas probable
        delta_t(i) = max_{q1,...,qt-1} P(q1,...,qt-1,O1,...,Ot,q_t=i|lambda)
        delta_t(i) = max_j [delta_t-1(j)a_ji]*log(P(o|q_t=i,lambda))
        A es la matriz de transiciones [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna delta_t(i) 
        """
        return np.max(delta+np.log(self.A.T),axis=1)+fb.probt[t]
    
    def viterbi(self,O,plot=True,maxg = False,absg = True,indexes=False): 
        """
        Algoritmo para determinar sucesion de estados mas probable
        A es la matriz de transiciones [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna sucesion de estados mas probable de forma ordenada usando el diccionario.
        """
        self.states_order(maxg=maxg,absg=absg)
        T =len(O)
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K)
        fb.prob_BN_t(O,self.sigma,self.P)
        delta = np.log(self.pi)+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T-self.P):
            delta = self.viterbi_step(delta,self.A,fb,i)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1-self.P):
            Q.append(psi[t][Q[t]])
        Q=np.array(Q)
        if indexes==False:
            [index,mag] = self.dictionary
            states = []
            for i in range(len(Q)):
                states.append(mag[index[Q[i]]])
            Q= np.array(states)
            
        if plot==True:
            plt.figure("Sequence of labeled hidden states of LMSAR")
            plt.title("Sequence of labeled hidden state of LMSAR")
            plt.ylabel("State magnitude" )
#            plt.ylabel("% Change respect to "+r"$||\mu_{\min}||_2 =$" +  str("{0:.2f}".format(np.min(np.sqrt(np.sum(np.array(self.mus)**2,axis=1))))))
            plt.xlabel("Time units")
            plt.plot(Q[::-1])
            plt.grid("on")
            plt.tight_layout()
#            plt.plot(Q)
            plt.show()
        return Q[::-1]
    
    
    def sample(self,leng,pseed,ini=None):
        """
        Generates a sample from the current parameters of the desired leng. 
        A pseed of length P*xK must be provided
        returns the generated data and the sequence of hidden states
        """
        Q = []
        data = pseed
        for t in range(leng):
            data_t = np.zeros(self.K)
            if t == 0:
                qt = np.argmax(np.random.multinomial(1,self.pi))
            else:
                qt = np.argmax(np.random.multinomial(1,self.A[qt]))
            Q.append(qt)
            for k in self.L[qt]:
                pak = self.my_fathers(self.G[qt],k)
                mean  = self.B[qt][k][0]
                if (len(pak) != 0):
                    mean = mean+ np.sum(data_t[pak]*self.B[qt][k][1:len(pak)+1])  
                if self.p[qt][k] != 0:
                    mean = mean+np.sum((data.T[k])[-self.p[qt][k]:]*self.B[qt][k][-self.p[qt][k]:])
                data_t[k] = np.random.normal(mean,self.sigma[qt][k])
            data_t = data_t[np.newaxis]
            data = np.concatenate([data,data_t],axis=0)
        return [data,Q]
     
    
    def act_mut(self,B,P):
        """
        actualiza mut de cada forback
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].mu_t(self.O[aux_len[i]:aux_len[i+1]],B,self.N,self.P,self.K)
    
    def act_gamma(self,A,sigma,pi):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward_As(A,sigma,pi,self.O[aux_len[i]:aux_len[i+1]],self.P)
    
       
    def act_A(self,A,sigma):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_aij(A,self.N,sigma,self.O[aux_len[0]:aux_len[1]],self.P)
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(A,self.N,sigma,self.O[aux_len[i]:aux_len[i+1]],self.P)
            numa = numa +self.forBack[i].numa 
            dena = dena +self.forBack[i].dena 
        A = np.ones([self.N,self.N])
        for j in range(self.N):
            A[j] = numa[j]/dena[j]
        return A
        
    def act_pi(self):
        """
        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        return api/len(self.lengths)
    
    def act_B(self,P):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_coef(self.O[aux_len[0]:aux_len[1]],self.P)
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(self.O[aux_len[i]:aux_len[i+1]],self.P)
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        B= []
        for i in range(self.N):
            Bi = []
            for k in range(self.K):
                if np.prod(bc[i][k].shape)> 1:
                    Bik = np.linalg.solve(bc[i][k],ac[i][k])
                else:
                    Bik = (ac[i][k]/bc[i][k])[0]
                Bi.append(Bik)
            B.append(Bi)
        return B
        
    def act_S(self):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_sigma(self.O[aux_len[0]:aux_len[1]],self.N,self.P)
        nums = self.forBack[0].numsig
        dens = self.forBack[0].densig
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_sigma(self.O[aux_len[i]:aux_len[i+1]],self.N,self.P)
            nums = nums+ self.forBack[i].numsig
            dens = dens+self.forBack[i].densig
        cov = []
        for k in range(self.K):
            covi = (nums[k]/dens)**0.5
            cov.append(covi)
        return np.array(cov)
    
    
    def pen(self,P,lengths):
        """
        Calculates the penalty of a list of graphs
        G is a list of graphs
        Returns a 2 index vector, where the first component is the penalty
            and the second is the number of parameters
        """
        T =np.sum(lengths)
        b= self.N+self.N**2+self.N*self.K+self.K+P*self.K*self.N
        return [-b*0.5*np.log(T),b]
    
    def EM(self,its=200,err=1e-10): #Puede que tenga errores, toca revisar con calma
        """
        Realiza el algoritmo EM
        G es una lista tal que G_i es un gafo para el estado i
        L es una lista, donde L_i es el ordenamiento topologico del grafo Gi.
            L_i Se usa para dar el orden del calculo del parametro mu.
        A prior de matriz de transicion
        pi prior de distribucion incial
        mu prior de medias, es una lista, donde mu_i es las medias de las variables
            para el i-esimo estado
        sigma prior de desviaciones estandar, es una lista, donde sigma_i es las 
            desviacion de las variables para el i-esimo estado
        O es el conjunto de observaciones, donde O[t] tiene longitud K, y K es el
            numero de variables.
        Retorna los parametros optimizados
        """
        likeli= -1e6
        eps =1
        it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K)
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]], self.sigma, self.P)
            self.forBack.append(fb)
        while eps >err and it<its:
            self.act_gamma(self.A,self.sigma,self.pi)
            self.A = self.act_A(self.A,self.sigma)
            self.pi = self.act_pi()
            self.B = self.act_B(self.P)
            self.act_mut(self.B,self.P)
            self.sigma = self.act_S()
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            self.LogLtrain = likelinew
            self.bic = -2*(likelinew + self.pen(self.P,self.lengths)[0])
            self.b = self.pen(self.P,self.lengths)[1]
            for i,fb in enumerate(self.forBack):
                fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]], self.sigma, self.P)
            eps = np.abs(likelinew-likeli)
            likeli = likelinew
            print("EM-Epsilon:" + str(eps)+" with BIC: "+ str(likeli))
            it = it+1   
        self.mvn_param()
        self.states_order()
        if eps<err:
            self.converged = True
        
                
            
    def report(self,root=None,name="CIG",labels=None):
        """
        Report all the models parameters    
        """
        if root == None:
            root = os.getcwd()
        F =open(root+"/param_"+name+".txt","w")
        F.write("AR_ASLG parameters:"+"\n"+"\n")
        F.write("B: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.B[i]) +"\n")
        F.write(r"sigma: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.sigma[i]) +"\n")
        F.write(r"G: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.G[i]) +"\n")
        F.write(r"L: "+"\n") 
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.L[i]) +"\n")
        F.write("pi: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.pi[i]) +"\n")
        F.write("A: "+"\n") 
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.A[i]) +"\n")
        F.write("p: "+"\n") 
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.p[i]) +"\n")
        F.write("Dictionary: "+"\n") 
        F.write(str(self.dictionary) +"\n")  
        F.close()
        self.plot_graph(root,labels)

    def save(self,root=None, name = "hmm_"+str(datetime.datetime.now())[:10]):
        """
        Saves the current model
        """
        if root == None:
            root = os.getcwd()
        itemlist = [self.N,self.K,self.P,self.A,self.pi,self.B,self.sigma,
                    self.mus,self.covs,self.dictionary,self.lengths,
                    self.O,self.b,self.bic,self.LogLtrain]
        if self.kappa is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".lmsar", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        loads a model, must be an ashmm file 
        """
        if rootr[-6:] != ".lmsar":
            return "The file is not an ashmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N = loaded[0]
        self.K = loaded[1]
        self.P = loaded[2]
        self.A = loaded[3]
        self.pi = loaded[4]
        self.B =  loaded[5]
        self.sigma = loaded[6]
        self.mus = loaded[7]
        self.covs = loaded[8]
        self.dictionary = loaded[9]
        self.lengths = loaded[10]
        self.O = loaded[11]
        self.b = loaded[12]
        self.bic = loaded[13]
        self.LogLtrain = loaded[14]
        if len(loaded)>15:
            self.nu = loaded[15]
            self.kappa = loaded[16]
    
    
class forback:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.mut = None
        self.ll = None
        self.probt = None
        self.numa = None
        self.dena = None
        self.matB = None
        self.coefmat = None
        self.coefvec =None
        self.numsig = None
        self.densig = None
        
    def mu_t(self,O,B,N,P,K): 
        """
        Calcula los valores de medias para las observaciones usando los valores de B
        O es un set de datos de K variables en T tiempos O[t]= [o_1,...,o_k]
        B es una lista que tiene coeficientes de cada estado, B_i =[B_i1,...,B_ik]
        L es el orden topologico, solo se usa para ordenar el calculo.
        G es una lista de grafos de cada estado G=[G_1,...,G_N]
        retorna mu_t una lista para los valore de mu para cada estado, cada variable y tiempo
        """
        T =len(O.T[0])
        mut = []
        for  i in range(N):
            mui = np.zeros([T-P,K])
            for k in range(K):
                acum = np.ones([T-P,1])
                for j in range(1,P+1):
                    x = np.roll(O.T[k],j)[P:]
                    x = x.reshape((len(x),1))
                    acum = np.concatenate([acum,x],axis=1)
                mui.T[k] = np.sum(acum*(B[i][k].T),axis=1)
            mut.append(mui)
        self.mut = np.array(mut)
    
    def prob_BN(self,o,mu,sigma):  #Este si toca cambiar 
        """
        Calcula [P(O_t|q_t=i,lambda)]_i
        o es una observacion, tiene longitud K, donde K es el numero de variables
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las desviaciones estandar para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        Retorna un vector de probabilidades de longitud N, donde N es el numero de
            estados
        """
        p = np.sum(-0.5*(np.log(2*np.pi)+np.log(sigma**2)+((o-mu)/sigma)**2),axis=1)
        # p = np.prod((np.sqrt(2*np.pi*sigma**2)**-1)*np.exp(-0.5*((o-mu)/sigma)**2),axis=1)
        return p

    
    def prob_BN_t(self,O,sigma,P):
        """
        Calcula [P(O_t|qt=i)]_ti
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las desviaciones estandar para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        t0 es un tiempo inicial
        t1 es un tiempo final
        Retorna una  matriz B, donde B[t][i] = P(Ot|qt=i)
        """
        T = len(O)
        B = []
        for t in range(T-P):
            B.append(self.prob_BN(O[t+P],self.mut[:,t,:],sigma))
        self.probt = np.array(B)
        
    def forward_backward_As(self,A,sigma,pi,O,P): 
        """
        ALgortimo Forward Backward escladao.
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        pi es vector de longitud N que determina la probabilidad de iniciar en estado i
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        Retorna 3 d-arrays, ALFA es una matriz, donde ALFA[t] es un vector de longitud N
            que determina el valor de la variable forward en el tiempo t, para los N estados.
            Similar sucede con la matriz BETA. Clist es el vector de escalamiento, 
            Clist permite calcular log-verosimilitudes.
        """
        T = len(O)
        alfa = np.log(pi)+ self.probt[0]
        Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
        Clist = np.array([Cd])
        Clist = np.array([Cd])
        alfa = alfa+Cd
        ALFA = [alfa]
        for t in range(1,T-P):
            alfa = self.forward_step_continuous(alfa,A,t)
            Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
            Clist = np.concatenate([Clist,[Cd]])
            alfa = Cd + alfa
            ALFA.append(alfa)
        ALFA= np.array(ALFA)
        self.alpha = ALFA
        
        beta =np.zeros(len(pi))
        Clist =Clist[::-1]
        beta = beta + Clist[0]
        BETA = [beta]
        for t in range(1,T-P):
            beta = self.backward_step_continuous(beta,A,T-t-P)
            beta = beta + Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.gamma = self.alpha +self.beta -np.max(self.alpha+self.beta,axis=1)[np.newaxis].T -np.log(np.sum(np.exp(self.alpha+self.beta-np.max(self.alpha+self.beta,axis=1)[np.newaxis].T),axis=1))[np.newaxis].T
        self.gamma = np.exp(self.gamma)

    
    def forward_step_continuous(self,alfa,A,t):
        """
        Hace un paso inductivo en la variabe alfa
        alfa_t(i) = P(O1,..., Ot, q_t=i|lambda)
        al_t(i) = sum_j alfa_(t-1)(j)*a_ji*P(O_t|q_t=i)
        alfa es un vector que represente alfa_(t-1)
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna alfa_t
        """
        maxi =np.max(alfa)
        logaa = np.log(np.dot(np.exp(alfa-maxi),A))
        return self.probt[t] +maxi+logaa
    
    def backward_step_continuous(self,beta,A,t):
        """
        Hace un paso inductivo en la variabe beta
        beta_t(i) = P(Ot+1,..., OT| q_t=i,lambda)
        beta_t(i) = sum_i beta_(t+1)(i)*a_ji*P(O_t+1|q_t=i)
        beta es un vector que represente beta_(t+1)
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna beta_t
        """
        maxi = np.max(beta)
        logba = np.log(np.dot(A,np.exp(self.probt[t]+beta-maxi)))
        return maxi+logba
        
    def act_aij(self,A,N,sigma,O,P): #TODO hacerlo de forma totalmente vectorial
        """
        Actualiza la matriz de transicion
        A es la matriz de transicion
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Retorna matriz de transicion actualizada
        """
        T = len(O)-P
        bj = self.probt
        nume = []
        deno = []
        for i in range(N):
            alfat = (self.alpha.T[i])[:T-1] 
            betat = self.beta[1:].T 
            num = A[i]*np.sum(np.exp(alfat+betat+ bj[1:].T),axis=1)
            den = np.sum(num)
            nume.append(num)
            deno.append(den)
        self.numa = np.array(nume)
        self.dena = np.array(deno)
        
    def act_coef(self,O,P): 
        """
        Actualiza los coeficientes de la cl de mu para cada variable
        G es una lista tal que G_i es un gafo para el estado i
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Retorna una lista B, donde Bi es una lista para el estado i, Bik posee los 
            coeficientes de mixtura de media de la variable k para el grafo i.
        """
        w = self.gamma
        N = len(w.T)
        K = len(O[0])
        T = len(O)
        bc = []
        ac = []
        
        acum = []
        for k in range(K):
            acumk = np.ones([1,T-P])
            for j in range(1,P+1):
                z = np.roll(O.T[k],j)[P:]
                z = z.reshape((1,len(z)))
                acumk = np.concatenate([acumk,z],axis=0)
            acum.append(acumk.T)
        acum = np.array(acum)
            
        for i in range(N):
            wi = w.T[i]
            ack = []
            bck = []
            for k in range(K):
                a = np.sum((wi*O.T[k][P:])[np.newaxis].T*acum[k],axis=0)
                b = wi[0]*np.outer(acum[k][0],acum[k][0])
                for t in range(1,T-P):
                    b +=  wi[t]*np.outer(acum[k][t],acum[k][t])
                bck.append(b)
                ack.append(a)
            bc.append(bck)
            ac.append(ack)
        self.coefmat = bc
        self.coefvec = ac
    

    def act_sigma(self,O,N,P):
        """
        Actualiza los valores de desviaciones estandar 
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Restorna una lista sigma, donde sigma_i son los valores de desviacion estandar
            para el estado i, sigma_ik es la desviacion de la variable k en el estado i
        """
        w= self.gamma
        K = len(O.T) 
        numk = np.zeros(K)
        denk = 0.0
        for k in range(K):
            for i in range(N):
                wi= w.T[i]
                numk = numk + np.sum(wi*((O[P:]-self.mut[i]).T)**2,axis=1)
                denk = denk + np.sum(wi)
        self.numsig = np.array(numk)
        self.densig = np.array(denk)
        
            
            
