# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:53:14 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import numpy as np
import time
import matplotlib.pyplot as plt 
import os
import math
import warnings
warnings.filterwarnings("ignore")
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
from KDE_AsHMM import KDE_AsHMM
from KDE_AsHMM import forBack
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
#%% Functions to generate synthetic data
def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs
                      if abs(sums[i] - sums[j]) <= hsd]
            # keep only longest

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]


    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2),
              (begin, distanceh - bigtick / 2)],
             linewidth=0.7)
        line([(end, distanceh + bigtick / 2),
              (end, distanceh - bigtick / 2)],
             linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD",
             ha="center", va="bottom")

        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=2.5)
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline), (end, cline)],
             linewidth=2.5)
        line([(begin, cline + bigtick / 2),
              (begin, cline - bigtick / 2)],
             linewidth=2.5)
        line([(end, cline + bigtick / 2),
              (end, cline - bigtick / 2)],
             linewidth=2.5)

    if filename:
        print_figure(fig, filename, **kwargs)
# Funciones para hacer parsing
def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd


def parse_results(ll, times,root,name,leng):
    means = np.mean(ll,axis=1)
    stds    = np.std(ll,axis=1)
    file = root+"\\"+name + ".txt"
    file2 = root+"\\"+name+"_xunit" + ".txt"
    f = open( file,"w")
    f.write(r"\begin{table}"+"\n")
    f.write(r"\centering"+"\n")
    f.write(r"\begin{tabular}{lrrr}"+"\n")
    f.write("Model        &  mean LL & std LL & Time(s)" + r"\\"+"\n")
    f.write(r" \hline"+"\n")
    f.write(r"KDE-HMM     &"+str(round(means[0],2))+ " & " +str(round(stds[0],2))+ "&" +str(round(times[0],2))+ r"\\"+" \n")
    f.write(r"KDE-AsHMM   &"+str(round(means[1],2))+ " & " +str(round(stds[1],2))+ "&" +str(round(times[1],2))+ r"\\"+"\n")
    f.write(r"KDE-ARHMM   &"+str(round(means[2],2))+ " & " +str(round(stds[2],2))+ "&" +str(round(times[2],2))+ r"\\"+"\n")
    f.write(r"KDE-BNHMM   &"+str(round(means[3],2))+ " & " +str(round(stds[3],2))+ "&" +str(round(times[3],2))+ r"\\"+"\n")
    f.write(r"HMM         &"+str(round(means[4],2))+ " & " +str(round(stds[4],2))+ "&" +str(round(times[4],2))+ r"\\"+"\n")
    f.write(r"AR-AsLG-HMM &"+str(round(means[5],2))+ " & " +str(round(stds[5],2))+ "&" +str(round(times[5],2))+ r"\\"+"\n")
    f.write(r"KDE-AsHMM*  &"+str(round(means[6],2))+ " & " +str(round(stds[6],2))+ "&" +str(round(times[6],2))+ r"\\"+"\n")
    f.write(r"\end{tabular}" +"\n")
    f.write(r"\caption{Mean likelihood, log likelihood standard deviation and viterbi error for all the compared models}"+"\n" )
    f.write(r"\label{table:synthetic_results}"+"\n" )
    f.write(r"\end{table}")
    f.close()
    
    means1   = np.mean(ll/leng,axis=1)
    stds1    = np.std(ll/leng,axis=1)
    f = open( file2,"w")
    f.write(r"\begin{table}"+"\n")
    f.write(r"\centering"+"\n")
    f.write(r"\begin{tabular}{lrrr}"+"\n")
    f.write("Model        &  mean LL & std LL & Time(s)" + r"\\"+"\n")
    f.write(r" \hline"+"\n")
    f.write(r"KDE-HMM     &"+str(round(means1[0],2))+ " & " +str(round(stds1[0],2))+ "&" +str(round(times[0],2))+ r"\\"+" \n")
    f.write(r"KDE-AsHMM   &"+str(round(means1[1],2))+ " & " +str(round(stds1[1],2))+ "&" +str(round(times[1],2))+ r"\\"+"\n")
    f.write(r"KDE-ARHMM   &"+str(round(means1[2],2))+ " & " +str(round(stds1[2],2))+ "&" +str(round(times[2],2))+ r"\\"+"\n")
    f.write(r"KDE-BNHMM   &"+str(round(means1[3],2))+ " & " +str(round(stds1[3],2))+ "&" +str(round(times[3],2))+ r"\\"+"\n")
    f.write(r"HMM         &"+str(round(means1[4],2))+ " & " +str(round(stds1[4],2))+ "&" +str(round(times[4],2))+ r"\\"+"\n")
    f.write(r"AR-AsLG-HMM &"+str(round(means1[5],2))+ " & " +str(round(stds1[5],2))+ "&" +str(round(times[5],2))+ r"\\"+"\n")
    f.write(r"KDE-AsHMM*  &"+str(round(means1[6],2))+ " & " +str(round(stds1[6],2))+ "&" +str(round(times[6],2))+ r"\\"+"\n")
    f.write(r"\end{tabular}" +"\n")
    f.write(r"\caption{Mean likelihood, log likelihood standard deviation and viterbi error for all the compared models}"+"\n" )
    f.write(r"\label{table:synthetic_results}"+"\n" )
    f.write(r"\end{table}")
    f.close()
    
    return [means,stds,means1,stds1]



def dag_v(G):
    """
    Executes the Kahn topological sorting  algorithm. 

    Parameters
    ----------
    G : TYPE numpy array size KxK
        DESCRIPTION. a graph with K variables

    Returns
    -------
    list
        DESCRIPTION. A bool indicating if it is a acyclic graph and a topological sort in L. 

    """
    G2 = np.copy(G)
    L = []
    booli = False
    ss = np.sum(G2,axis=1)
    s = np.where(ss==0)[0]
    while len(s)>0:
        si = s[0]
        s = s[1:]
        L.append(si)
        indexi = np.where(G2.T[si]==1)[0]
        for x in indexi:
            G2[x][si] = 0
            if np.sum(G2[x])==0:
                s =np.concatenate([s,[x]])
    if np.sum(G2)==0:
        booli = True
    return [booli,np.array(L)]

# Helpful functions

def square(x):
    return x**2

def sin(x):
    return np.sin(2*np.pi*2*x)

def ide(x):
    return x

def log_gaussian_kernel(x):
    return -0.5*(x**2+np.log(2*np.pi))

def gaussian_kernel(x):
    return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

# Generate multidimensional data

def gen_nl_random(ns,seq,G,L,P,p,M,means,sigma,k,f1,f2):
    ncols = len(means[0])
    x = np.zeros([P,ncols])
    for l in seq:
        xl = gen_nl_random_l(ncols, ns[l], G[l], L[l], P, p[l], M[l], means[l], sigma[l], f1,f2,k[l],x)
        x = np.concatenate([x,xl],axis=0)
    return x[P:]


def gen_nl_random_l(ncols,nrows,G,L,P,p,M,means,sigma,f1,f2,k,y):
    x = y[-P:]
    for t in range(nrows):
        xt = gen_nl_random_1sample(G, L, P, p, M, means, sigma, f1,f2,k, x[-P:])
        x = np.concatenate([x,xt],axis=0)
    return x[P:]


def gen_nl_random_1sample(G,L,P,p,M,means,sigma,f1,f2,k,seed ):
    K = len(G)
    xt = np.ones([1,K])
    for i in L:
        if np.sum(M[i]) ==0 :
            xt[0][i] = np.random.normal(means[i],sigma[i],1)
        else:
            mean  = means[i]
            for w in range(K):
                mean += M[i][w]*f1(means[w]-xt[0][w])+k[i]
            for l in range(P):
                f2m = M[i][K+l]*f2((seed[-l-1][i]))
                mean += f2m
            xt[0][i] = np.random.normal(mean,sigma[i],1)
    return xt

def load_model(j):
    srootj = sroot+"\\"+"models_len"+str(j)
    data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
    model1 = KDE_AsHMM(data_gen, 3,P=P)
    model2 = KDE_AsHMM(data_gen, 3,P=P)
    model21 = KDE_AsHMM(data_gen, 3,P=P,struc=False)
    model22 = KDE_AsHMM(data_gen, 3,P=P,lags=False)
    model3 = hmm(data_gen,lengths_gen,3,P=P)
    model4 = hmm(data_gen, lengths_gen,3,P=P)
    model5 = KDE_AsHMM(data_gen,3,p=p,G=G,P=P)
    model1.load(srootj  + "\\" + "synt_mod1_"+str(j)+".kdehmm")
    model2.load(srootj  + "\\" + "synt_mod2_"+str(j)+".kdehmm")
    model21.load(srootj + "\\" + "synt_mod21_"+str(j)+".kdehmm")
    model22.load(srootj + "\\" + "synt_mod22_"+str(j)+".kdehmm")
    model3.load(srootj  + "\\" + "synt_mod3_"+str(j)+".ashmm")
    model4.load(srootj  + "\\" + "synt_mod4_"+str(j)+".ashmm")
    model5.load(srootj  + "\\" + "synt_mod5_"+str(j)+".kdehmm")
    return [model1,model2,model21,model22,model3,model4,model5]
    
            
#%% Generating data
K       = 7
N       = 3
nses    = [350,350,350]
seqss   = [0,1,2,1,0,2,0]

G = np.zeros([3,7,7])
G[1][1][0] = 1; G[1][2][0] = 1; G[1][4][3] = 1
G[2][1][0] = 1; G[2][2][0] = 1; G[2][4][3] = 1;  G[2][3][0]  =1; G[2][4][0]  =1

k = np.zeros([N,K])

L          = []
for i in range(3):
    lel = dag_v(G[i])
    print(lel[0])
    L.append(lel[1])

L            = np.array(L)
P            = 2
p            = np.array([[0,0,0,0,0,0,0],
                         [0,1,0,1,0,0,0],
                         [1,2,1,2,1,0,0]])

means_g = np.array([ [    0,    -10,  20,     0,    8, 1.0, 2.0],
                     [    2,    -1,     3,     2,    2, 1.0, 2.0], 
                     [   -2,     2,    -3,    -2,   -2, 1.0, 2.0]])

sigmas_g = np.array( [[0.1,   0.3,  0.5,  0.2,  1.8, 0.5, 0.6],
                      [1,     0.7,  0.2,  1.2,  1.4, 0.5, 0.6], 
                      [2,     0.5,  0.6,  1.3,  0.2, 0.5, 0.6]])

MT            = np.zeros([N,K,K+P])
MT[1][1][0] = 1.5; MT[1][2][0] =  -0.9; MT[1][4][3] = 2 
MT[2][1][0] = -0.9; MT[2][2][0] = 1.5; MT[2][4][3] = -2 

MT[1][1][7]   =  0.5; MT[1][3][7] = 0.6 
MT[2][0][7]   =  0.4; MT[2][1][7] = 0.4; MT[2][1][8] = 0.4; MT[2][2][7] =  0.4 ; MT[2][3][7] = -0.5 ; MT[2][3][8] = -0.3;  MT[2][4][7]= 0.6

data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
lengths_gen  = np.array([len(data_gen)])
sroot = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\models"
#%% Generating several models
test_nlen = [50,100,150,200,250,300,350]
train = False
n_pruebas = 300
means  = []
stds = []
meansx = []
stdsx = []
logs = []
nss = (np.ones(3)*200).astype(int)
testing_data = []
for t in range(n_pruebas):
    testing_data.append(gen_nl_random(nss,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide))

#%%
for j in test_nlen:
    try:
        os.mkdir(sroot+"\\"+"models_len"+str(j))
    except:
        pass
    srootj = sroot+"\\"+"models_len"+str(j)
    nses = (np.ones(3)*j).astype(int)
    
    tr = np.repeat(np.array(seqss),j)
    depo = np.ones([j*len(seqss),3])*1e-18
    
    for i in range(3):
        indexi = np.argwhere(tr==i).T[0]
        depo[indexi,i] = 1 
    depo = depo/np.sum(depo,axis=0)
    depo = depo[P:]
    
    data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
    lengths_gen  = np.array([len(data_gen)])
    times = []
    model1 = KDE_AsHMM(data_gen, 3,P=P)
    model2 = KDE_AsHMM(data_gen, 3,P=P)
    model21 = KDE_AsHMM(data_gen, 3,P=P,struc=False)
    model22 = KDE_AsHMM(data_gen, 3,P=P,lags=False)
    model3 = hmm(data_gen,lengths_gen,3,P=P)
    model4 = hmm(data_gen, lengths_gen,3,P=P)
    model5 = KDE_AsHMM(data_gen,3,p=p,G=G,P=P,v=depo)
    if train == True:
        tick1 = time.time()
        model1.EM()
        tock1 = time.time()
        
        tick2 = time.time()
        model2.SEM()
        tock2 = time.time()
        
        tick3 = time.time()
        model21.SEM()
        tock3 = time.time()
        

        tick4 = time.time()
        model22.SEM()
        tock4 = time.time()
        
        tick5 = time.time()
        model3.EM()
        tock5 = time.time()
        
        try:
            tick6 = time.time()
            model4.SEM()
            tock6 = time.time()
        except:
            model4 = hmm(data_gen, lengths_gen,3,P=P)
            tick6 = time.time()
            model4.EM()
            tock6 = time.time()
        
        tick7 = time.time()
        model5.EM()
        tock7 = time.time()
        
        model1.save(srootj,name="synt_mod1_"+str(j))
        model2.save(srootj,name="synt_mod2_"+str(j))
        model21.save(srootj,name="synt_mod21_"+str(j))
        model22.save(srootj,name="synt_mod22_"+str(j))
        model3.save(srootj,name="synt_mod3_"+str(j))
        model4.save(srootj,name="synt_mod4_"+str(j))
        model5.save(srootj,name="synt_mod5_"+str(j))
        times = [tock1-tick1,tock2-tick2,tock3-tick3, tock4-tick4, tock5-tick5, tock6-tick6, tock7-tick7]
        np.save(srootj+"\\"+"tiempos_long", times)
    else:
        model1.load(srootj  + "\\" + "synt_mod1_"+str(j)+".kdehmm")
        model2.load(srootj  + "\\" + "synt_mod2_"+str(j)+".kdehmm")
        model21.load(srootj + "\\" + "synt_mod21_"+str(j)+".kdehmm")
        model22.load(srootj + "\\" + "synt_mod22_"+str(j)+".kdehmm")
        model3.load(srootj  + "\\" + "synt_mod3_"+str(j)+".ashmm")
        model4.load(srootj  + "\\" + "synt_mod4_"+str(j)+".ashmm")
        model5.load(srootj  + "\\" + "synt_mod5_"+str(j)+".kdehmm")
        times = np.load(srootj+"\\"+"tiempos_long.npy")

    pruebas = n_pruebas
    ll1  = []
    ll2  = []
    ll21 = []
    ll22 = []
    ll3  = []
    ll4  = []
    ll5  = []
    error = []
    for t in range(pruebas):
        data_gen_test = testing_data[t]
        ll1.append(model1.log_likelihood(data_gen_test))
        ll2.append(model2.log_likelihood(data_gen_test ))
        ll21.append(model21.log_likelihood(data_gen_test ))
        ll22.append(model22.log_likelihood(data_gen_test ))
        ll3.append(model3.log_likelihood(data_gen_test ))
        ll4.append(model4.log_likelihood(data_gen_test ))
        ll5.append(model5.log_likelihood(data_gen_test ))
        print(str(round(100*(t+1)/(pruebas*1.),3))+"%")
    ll1  = np.array(ll1)
    ll1  = ll1[~np.isnan(ll1)]
    ll2  = np.array(ll2)
    ll2  = ll2[~np.isnan(ll2)]
    ll21 = np.array(ll21)
    ll21 = ll21[~np.isnan(ll21)]
    ll22 = np.array(ll22)
    ll22 = ll22[~np.isnan(ll22)]
    ll3  = np.array(ll3)
    ll3  = ll3[~np.isnan(ll3)]
    ll4  = np.array(ll4)
    ll4  = ll4[~np.isnan(ll4)]
    ll5  = np.array(ll5)
    ll5  = ll5[~np.isnan(ll5)]
    
    print("Likelihood KDE-HMM:               "+ str(np.mean(ll1)))
    print("Likelihood KDE-AsHMM:             "+ str(np.mean(ll2)))
    print("Likelihood KDE-AsHMM no BN opt:   "+ str(np.mean(ll21)))
    print("Likelihood KDE-AsHMM no AR opt:   "+ str(np.mean(ll22)))
    print("Likelihood HMM:                   "+ str(np.mean(ll3)))
    print("Likelihood AR-AsLG-HMM:           "+ str(np.mean(ll4)))
    print("Likelihood KDE-HMM with known BN: "+ str(np.mean(ll5)))
    ll =  np.array([ll1, ll2, ll21, ll22, ll3, ll4, ll5])
    rr = parse_results(ll,times,r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt","syn_table_"+str(j),len(data_gen_test))
    means.append(rr[0])
    stds.append(rr[1])
    meansx.append(rr[2])
    stdsx.append(rr[3])
    logs.append(ll)
    print("Temine para T_train = "+str(j))
means  = np.array(means)
stds   = np.array(stds)
meansx = np.array(meansx)
stdsx  = np.array(stdsx)
logs   = np.array(logs)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\meansx",meansx)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\stdsx",stdsx)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\means",means)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\stds",stds)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\logs",logs)
#%% Load results
meansx = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\meansx.npy")
stdsx  = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\stdsx.npy")
means  = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\means.npy")
stds   = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\stds.npy")
logs   = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\logs.npy")
#%% plots
plt.figure("Mean likelihood ")
plt.clf()
plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,0],label = "KDE-HMM",color="red",linestyle= "dotted")
# plt.fill_between(np.arange(1,7)*350,means[:,0]+2*stds[:,0],means[:,0]-2*stds[:,0],color="red",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,1],label = "KDE-AsHMM",color="blue",linestyle = "solid")
# plt.fill_between(np.arange(1,7)*350,means[:,1]+2*stds[:,1],means[:,1]-2*stds[:,1],color="blue",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,2],label = "KDE-ARHMM",color="green", linestyle = (0,(5,1)))
# plt.fill_between(np.arange(1,7)*350,means[:,2]+2*stds[:,2],means[:,2]-2*stds[:,2],color="green",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,3],label = "KDE-BNHMM",color="magenta",linestyle = "dashed")
# plt.fill_between(np.arange(1,7)*350,means[:,3]+2*stds[:,3],means[:,3]-2*stds[:,3],color="gray",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,4],label = "HMM",color="black",linestyle = "dashdot")
# plt.fill_between(np.arange(1,7)*350,means[:,4]+2*stds[:,4],means[:,4]-2*stds[:,4],color="black",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,5],label = "AR-AsLG-HMM",color="orange", linestyle = (0,(1,2)))
# plt.fill_between(np.arange(1,7)*350,means[:,5]+2*stds[:,5],means[:,5]-2*stds[:,5],color="orange",alpha=0.5)

plt.plot(np.arange(1,len(test_nlen)+1)*350,means[:,6],label = "KDE-AsHMM*",color="cyan", linestyle = (0,(2,2)))
# plt.fill_between(np.arange(1,7)*350,means[:,5]+2*stds[:,5],means[:,5]-2*stds[:,5],color="orange",alpha=0.5)
plt.grid("on")
plt.ylabel("$\mu(LL)$")
plt.xlabel("$T$ for training")
plt.legend()


plt.figure("std likelihood per unit data ")
plt.clf()
plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,0],label = "KDE-HMM",color="red",linestyle= "dotted",linewidth=2)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,0]+2*stdsx[:,0],meansx[:,0]-2*stdsx[:,0],color="red",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,1],label = "KDE-AsHMM",color="blue",linestyle = "solid")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,1]+2*stdsx[:,1],meansx[:,1]-2*stdsx[:,1],color="blue",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,2],label = "KDE-ARHMM",color="green", linestyle = (0,(5,1)))
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,2]+2*stdsx[:,2],meansx[:,2]-2*stdsx[:,2],color="green",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,3],label = "KDE-BNHMM",color="magenta",linestyle = "dashed")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,3]+2*stdsx[:,3],meansx[:,3]-2*stdsx[:,3],color="magenta",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,4],label = "HMM",color="black",linestyle = "dashdot")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,4]+2*stdsx[:,4],meansx[:,4]-2*stdsx[:,4],color="black",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,5],label = "AR-AsLG-HMM",color="orange", linestyle = (0,(1,2)),linewidth=3)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,5]+2*stdsx[:,5],meansx[:,5]-2*stdsx[:,5],color="orange",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,stdsx[:,6],label = "KDE-AsHMM*",color="gray", linestyle = (0,(3,1)),linewidth=3)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,5]+2*stdsx[:,5],meansx[:,5]-2*stdsx[:,5],color="cyan",alpha=0.1)

plt.grid("on")
plt.ylabel("$\sigma(LL/T_{test})$")
plt.xlabel("$T$ for training")
plt.legend()
plt.tight_layout()



plt.figure("Mean likelihood per unit data ")
plt.clf()
plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,0],label = "KDE-HMM",color="red",linestyle= "dotted",linewidth=2)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,0]+2*stdsx[:,0],meansx[:,0]-2*stdsx[:,0],color="red",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,1],label = "KDE-AsHMM",color="blue",linestyle = "solid")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,1]+2*stdsx[:,1],meansx[:,1]-2*stdsx[:,1],color="blue",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,2],label = "KDE-ARHMM",color="green", linestyle = (0,(5,1)))
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,2]+2*stdsx[:,2],meansx[:,2]-2*stdsx[:,2],color="green",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,3],label = "KDE-BNHMM",color="magenta",linestyle = "dashed")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,3]+2*stdsx[:,3],meansx[:,3]-2*stdsx[:,3],color="magenta",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,4],label = "HMM",color="black",linestyle = "dashdot")
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,4]+2*stdsx[:,4],meansx[:,4]-2*stdsx[:,4],color="black",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,5],label = "AR-AsLG-HMM",color="orange", linestyle = (0,(1,2)),linewidth=3)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,5]+2*stdsx[:,5],meansx[:,5]-2*stdsx[:,5],color="orange",alpha=0.1)

plt.plot(np.arange(1,len(test_nlen)+1)*350,meansx[:,6],label = "KDE-AsHMM*",color="gray", linestyle = (0,(3,1)),linewidth=3)
# plt.fill_between(np.arange(1,len(test_nlen)+1)*350,meansx[:,5]+2*stdsx[:,5],meansx[:,5]-2*stdsx[:,5],color="cyan",alpha=0.1)

plt.grid("on")
plt.ylabel("$\mu(LL/T_{test})$")
plt.xlabel("$T$ for training")
plt.legend()
plt.tight_layout()

#%% compute the rankings
from scipy.stats import chi2 as chi

# logs = np.array(logs)
# np.save(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\logs",logs)
logs  = np.load(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\synt\logs.npy")
rank_6 = []
pruebas = 0
for j in range(n_pruebas):
    rank_6.append(np.argsort(np.argsort(-logs[6,:,j])))
    pruebas = pruebas + 1 
rank_6 = np.array(rank_6)
avg_rank_6 = np.mean(rank_6,axis=0)

rank_0 = []
pruebas = 0
for j in range(n_pruebas):
    rank_0.append(np.argsort(np.argsort(-logs[0,:,j])))
    pruebas = pruebas + 1 
rank_0 = np.array(rank_0)
avg_rank_0 = np.mean(rank_0,axis=0)

# Friedman test
Q_0 = 12*pruebas/(7*8)*np.sum((avg_rank_0-4)**2)
Q_6 = 12*pruebas/(7*8)*np.sum((avg_rank_6-4)**2)
crit = chi.ppf(0.95,6)
p_val_0 = 1-chi.cdf(Q_0,6)
p_val_6 = 1-chi.cdf(Q_6,6)


cd = compute_CD(avg_rank_6, pruebas,alpha="0.05", test="nemenyi") #tested on 14 datasets 
names =["KDE-HMM","KDE-AsHMM","KDE-ARHMM","KDE-BNHMM","HMM","AR-AsLG-HMM","KDE-AsHMM*"]
graph_ranks(avg_rank_6, names, cd=cd, width=5, textspace=1.5)

