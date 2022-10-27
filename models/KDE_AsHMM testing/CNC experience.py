# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:05:11 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import math
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
import pandas as pd
from KDE_AsHMM import KDE_AsHMM as kde
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
from scipy.stats import chi2 as chi
#%% Funciones
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

def create_verdad(labels,llave):
    T = len(labels)
    N = len(llave)
    verdad  = np.ones([T,N])*1e-5
    for t in range(T):
        for i in range(len(llave)):
            if llave[i] == labels[t]:
                verdad[t][i] = 1
    return verdad
    
    

#%% Import data
# Download the files from the dataset https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill

# rootf = "PATH OF THE FILE WERE YOU UNZIPPED THE FILES"
rootf = r"C:\Users\fox_e\OneDrive\Documentos\datasets\CNC milling machine\CSV"
files = os.listdir(rootf)
datasets = []
verdad = []
P_u=1
llaves = {0: 'Layer 1 Down',
          1: 'Layer 1 Up',
          2: 'Layer 2 Up',
          3: 'Layer 2 Down',
          4: 'Layer 3 Down',
          5: 'Layer 3 Up',
          6: 'Repositioning'}


for f in files:   
    dataf = pd.read_csv(rootf+"\\"+f)
    dataf = dataf.drop(dataf[dataf["Machining_Process"] == "Prep"].index )
    dataf = dataf.drop(dataf[dataf["Machining_Process"] == "End"].index  )
    dataf = dataf.drop(dataf[dataf["Machining_Process"] == "end"].index  )
    dataf = dataf.drop(dataf[dataf["Machining_Process"] == "Starting"].index  )
    # dataf = dataf.drop(dataf[dataf["Machining_Process"] == "Repositioning"].index  )
    
    key = 'Machining_Process'
    xdif = np.array(dataf["X1_ActualPosition"])[np.newaxis]
    xdif = xdif-np.min(xdif)
    ydif = np.array(dataf["Y1_ActualPosition"])[np.newaxis]
    ydif = ydif-np.min(ydif)
    zdif = np.array(dataf["Z1_ActualPosition"])[np.newaxis]
    zdif = zdif-np.min(zdif)
    sdif = np.array(dataf["S1_ActualPosition"])[np.newaxis]
    sdif = sdif-np.min(sdif)

    datasetf = np.concatenate([xdif, ydif, zdif, sdif],axis=0).T
    datasetf = datasetf+np.random.normal(0,0.5,datasetf.shape)
    datasets.append(datasetf)
    verdadf = create_verdad(np.array(dataf[key]),llaves)
    verdad.append(verdadf)
labels_cnc= ["X-ActualPosition","Y-ActualPosition","Z-ActualPosition","Spindle-ActualPosition"]

worn  =  [3, 4, 5, 6, 7,  8,  9,  15]
unworn = [0, 1, 2, 10, 11, 12, 13, 14, 16,17]
train = True

n_states = len(llaves) #Number for hidden states for all models
# Folds definition
fold1 = [0,12] 
fold2 = [1,13]
fold3 = [2,14]
fold4 = [10,16]
fold5 = [11,17]
folds = [fold1,fold2,fold3,fold4,fold5]
#%% Data visualization
model = kde(np.concatenate([datasets[10],datasets[16]],axis=0),int(n_states),P=1,v=np.concatenate([verdad[10],verdad[16]],axis=0),)
model.plot_data_scatter()
model.plot_all_pairs_scatter()
#%% Training 
# root_accepted = "PATH WHERE YOU PUT WANT TO SAVE YOUR RESULTS"
root_accepted = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\accepted_models"
try:
    os.mkdir(root_accepted)
except:
    pass
models_accepted = []
for i in range(len(folds)):
    dataset_i = np.zeros([0,datasetf.shape[1]])
    verdadi = np.zeros([0,len(llaves)])
    for j in folds[i]:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
        verdadi = np.concatenate([verdadi,verdad[j]])
    verdadi = verdadi/np.sum(verdadi,axis=0)
    verdadi = verdadi[P_u:]
    modeli = []
    model1 = kde(dataset_i,n_states,P=P_u,v=verdadi,v_train=False)
    model2 = kde(dataset_i,n_states,P=P_u,v=verdadi,v_train=False)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    if train == True:
        print("######################")
        print("Begin Train of fold: "+str(i+1))
        print("######################")
        model1.EM()
        model1.save(root_accepted, name= "KDE_HMM_accepted"+str(i))
        model2.SEM()
        model2.save(root_accepted, name= "KDE_AsHMM_accepted" +str(i))
        model3.EM()
        model3.save(root_accepted, name= "HMM_accepted" +str(i))
        try:
            model4.SEM()
            model4.save(root_accepted, name= "AR-AsLG-HMM_accepted"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=P_u)
            model4.EM()
            model4.save(root_accepted, name= "AR-AsLG-HMM_accepted" +str(i))
        modeli.append([i,model1,model2,model3,model4])
    else:
        model1.load(root_accepted+"\\"+"KDE_HMM_accepted"+str(i)+".kdehmm")
        model2.load(root_accepted+"\\"+"KDE_AsHMM_accepted"+str(i)+".kdehmm")
        model3.load(root_accepted+"\\"+"HMM_accepted"+str(i)+".ashmm")
        model4.load(root_accepted+"\\"+"AR-AsLG-HMM_accepted"+str(i)+".ashmm")
        modeli.append([i,model1,model2,model3,model4])
    models_accepted.append(modeli)
    
#%% Testing
ll_accepted = []
for l in range(len(folds)):
    llacceptedi = []
    models_unworni = models_accepted[l]
    actual_i = models_unworni[0][0]
    mod1 = models_unworni[0][1]
    mod2 = models_unworni[0][2]
    mod3 = models_unworni[0][3]
    mod4 = models_unworni[0][4]
    for j in range(len(folds)):
        if j!= l:
            for k in folds[j]:
                ll1ij = mod1.log_likelihood(datasets[k],xunit=True)
                ll2ij = mod2.log_likelihood(datasets[k],xunit=True)
                ll3ij = mod3.log_likelihood(datasets[k],xunit=True)
                ll4ij = mod4.log_likelihood(datasets[k],xunit=True)
                llacceptedi.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_accepted.append(llacceptedi)
ll_accepted = np.array(ll_accepted)
np.save(root_accepted+"\\"+"ll_accepted",ll_accepted)

ll_accepted2non = []
for l in range(len(folds)):
    llacceptednoni = []
    models_unworni = models_accepted[l]
    actual_i = models_unworni[0][0]
    mod1 = models_unworni[0][1]
    mod2 = models_unworni[0][2]
    mod3 = models_unworni[0][3]
    mod4 = models_unworni[0][4]
    for j in worn:
        ll1ij = mod1.log_likelihood(datasets[j],xunit=True)
        ll2ij = mod2.log_likelihood(datasets[j],xunit=True)
        ll3ij = mod3.log_likelihood(datasets[j],xunit=True)
        ll4ij = mod4.log_likelihood(datasets[j],xunit=True)
        llacceptednoni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_accepted2non.append(llacceptednoni)
ll_accepted2non = np.array(ll_accepted2non)
np.save(root_accepted+"\\"+"ll_accepted2non",ll_accepted2non)

#%% Plots y tablas a mostrar
ll_accepted2non = np.load(root_accepted+"\\"+"ll_accepted2non.npy")
ll_accepted     = np.load(root_accepted+"\\"+"ll_accepted.npy")

# Mean log likelihoods per fold
mtll_accepted = np.mean(ll_accepted,axis=1)
mtll_accepted2non = np.mean(ll_accepted2non,axis=1)

# Mean log-likelihoods in average
fm_accepted     = np.mean(mtll_accepted,axis=0)   
fm_accepted2non = np.mean(mtll_accepted2non,axis=0)    

# Test de hipotesis
ranks = np.argsort(np.argsort(-ll_accepted))
ranki = np.zeros(4)
n_ranks = 0
for i in range(len(folds)):
    for j in range(len(worn)):
        ranki = ranki + ranks[i,j]
        n_ranks = n_ranks +1
avg_rank = ranki/n_ranks

# Test de Friedman
Q = 12*n_ranks/(4*5)*np.sum((avg_rank-5/2)**2)
p_val = 1-chi.cdf(Q,3)

# Test de Nemenyi
cd = compute_CD(avg_rank, n_ranks,alpha="0.05", test="nemenyi") #tested on 14 datasets 
names =["KDE-HMM","KDE-AsHMM","HMM","AR-AsLG-HMM"]
graph_ranks(avg_rank, names, cd=cd, width=5, textspace=1.5)    

#%% Para borrar
root_accepted = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models"
try:
    os.mkdir(root_accepted)
except:
    pass
models_accepted = []
for i in range(len(folds)):
    dataset_i = np.zeros([0,datasetf.shape[1]])
    for j in folds[i]:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
    modeli = []
    model1 = kde(dataset_i,n_states,P=P_u)
    model2 = kde(dataset_i,n_states,P=P_u)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model1.load(root_accepted+"\\"+"KDE_HMM_unworn"+str(i)+".kdehmm")
    model2.load(root_accepted+"\\"+"KDE_AsHMM_unworn"+str(i)+".kdehmm")
    model3.load(root_accepted+"\\"+"HMM_unworn"+str(i)+".ashmm")
    model4.load(root_accepted+"\\"+"AR-AsLG-HMM_unworn"+str(i)+".ashmm")
    modeli.append([i,model1,model2,model3,model4])
    models_accepted.append(modeli)
    

            
            
            

            
            



