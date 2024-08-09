import copy
import logging
import os
import time

from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy

from matplotlib.lines import Line2D
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy import signal
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils.mqtt_read import MQTT

from EQTransformer.core.EqT_utils import FeedForward, LayerNormalization, SeqSelfAttention, f1


from datetime import datetime
import shutil
import time


matplotlib.use('agg')

# CONFIGS = {
#     'broker': "202.52.240.148",
#     'port'  : 5065,
#     'topic' : 'outTopic',
# }

CONFIGS = {
    'broker': "100.100.200.120",
    'port'  : 1883,
    'topic' : 'topic',
}

EQT_VERSION = "0.1.61"
Q_MAXSIZE = 2400    # 60 * 40
SAMPLE_RATE = 40    # Hz
N_SAMPLES_LMT = 2400
N_SAMPLES_REQ = 6000    # 60 * 100


class DataTransform: 
    def __init__(self, norm_mode='std') -> None:
        self.norm_mode = norm_mode

    def _normalize(self, data: np.ndarray, mode='max'):          
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def transform(self, data_lst: list[np.ndarray]):
        if not isinstance(data_lst, list) and isinstance(data_lst, np.ndarray):
            data_lst = [data_lst]
        X = np.zeros((len(data_lst), N_SAMPLES_REQ, 3))
        for idx, data in enumerate(data_lst):
            # data = self._normalize(data, self.norm_mode)
            X[idx, :, :] = data
        return {'input': X}
    

    
# get all data from stream
def stream2array(streams: list[Stream]):
    data = []
    for st in streams:
        npz_data = np.zeros([N_SAMPLES_REQ, 3])  
        chanL = [ch.upper() if (ch:=tr.stats.channel[-1].strip()).isalpha() else ch for tr in st]
        
        if 'Z' in chanL:
            npz_data[:,2] = st[chanL.index('Z')].data[:N_SAMPLES_REQ]
        if ('E' in chanL) or ('1' in chanL):    
            try: 
                npz_data[:,0] = st[chanL.index('E')].data[:N_SAMPLES_REQ]
            except Exception:
                npz_data[:,0] = st[chanL.index('1')].data[:N_SAMPLES_REQ]
        if ('N' in chanL) or ('2' in chanL):        
            try: 
                npz_data[:,1] = st[chanL.index('N')].data[:N_SAMPLES_REQ]
            except Exception:
                npz_data[:,1] = st[chanL.index('2')].data[:N_SAMPLES_REQ]
        
        data.append(npz_data)
    
    return data


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _picker(args, yh1, yh2, yh3):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
   
    Returns
    -------    
    matches : dic
        Contains the information for the detected and picked event.            
        
    matches : dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
        
    yh3 : 1D array             
        normalized S_probability                              
                
    """                
             
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
    # print(f"{detection=}")  # Nested List of trigger on and of times in samples
    # print(f"{pp_arr=}") # indeces of the peaks in `x`.
    # print(f"{ss_arr=}") # indeces of the peaks in `x`.
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None  
            
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]

            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})                 
                
    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})             
            
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):                                 
                    
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})

    # print(f"{EVENTS=}")            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]    
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val}) 
             
            if len(candidate_Ss) > 1:                            
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val}) 
                else:         
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:                 
                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                                                
    return matches, pick_errors, yh3


def _load_model(
    model_path: str,
    loss_weights=[0.03, 0.40, 0.58],
    loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
    ):
    logging.info(f"*** Loading the model ...")
    
    model = load_model(
        model_path, 
        custom_objects={
            'SeqSelfAttention': SeqSelfAttention, 
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization, 
            'f1': f1                                                                            
        }
    )

    logging.info(f"*** Model Loading is complete!")
    
    model.compile(
        loss = loss_types,
        loss_weights = loss_weights,           
        optimizer = Adam(lr = 0.001),
        metrics = [f1]
    )
    return model


def reshape(ts_data: deque):
    if (n_ts_data:=len(ts_data)) < Q_MAXSIZE:
        n_deficient = Q_MAXSIZE - n_ts_data
        deficient_ele = [0] * n_deficient
        ts_data.extendleft(deficient_ele)
        data = np.array(ts_data)
    else:
        ts_data_lst = list(ts_data)
        data = ts_data_lst[:Q_MAXSIZE]
        data = np.array(data)
    
    return data.reshape(Q_MAXSIZE, -1)


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:  
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st


def _plotter_prediction(data, args, save_figs, yh1, yh2, yh3, evi, matches):

    """ 
    
    Generates plots of detected events with the prediction probabilities and arrival picks.

    Parameters
    ----------
    data: NumPy array
        3 component raw waveform.

    evi: str
        Trace name.  

    args: dic
        A dictionary containing all of the input parameters. 

    save_figs: str
        Path to the folder for saving the plots. 

    yh1: 1D array
        Detection probabilities. 

    yh2: 1D array
        P arrival probabilities. 
        
    yh3: 1D array
        S arrival probabilities.  

    matches: dic
        Contains the information for the detected and picked event. 
                  
        
    """  

    font0 = {'family': 'serif',
            'color': 'white',
            'stretch': 'condensed',
            'weight': 'normal',
            'size': 12,
            } 
   
    spt, sst, detected_events = [], [], []
    for match, match_value in matches.items():
        detected_events.append([match, match_value[0]])
        if match_value[3]: 
            spt.append(match_value[3])
        else:
            spt.append(None)
            
        if match_value[6]:
            sst.append(match_value[6])
        else:
            sst.append(None)    
            
    if args['plot_mode'] == 'time_frequency':
    
        fig = plt.figure(constrained_layout=False)
        widths = [6, 1]
        heights = [1, 1, 1, 1, 1, 1, 1.8]
        spec5 = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths,
                              height_ratios=heights, left=0.1, right=0.9, hspace=0.1)
        
        
        ax = fig.add_subplot(spec5[0, 0])         
        plt.plot(data[:, 0], 'k')
        plt.xlim(0, 6000)
        x = np.arange(6000)
        # if platform.system() == 'Windows':
        #     plt.title(save_figs.split("\\")[-2].split("_")[0]+":"+str(evi))
        # else:
        #     plt.title(save_figs.split("/")[-2].split("_")[0]+":"+str(evi))
                     
        ax.set_xticks([])
        plt.rcParams["figure.figsize"] = (10, 10)
        legend_properties = {'weight':'bold'} 
        
        pl = None
        sl = None            
        
        if len(spt) > 0 and np.count_nonzero(data[:, 0]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 0]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
        
    
        ax = fig.add_subplot(spec5[0, 1])                 
        if pl or sl: 
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')
        
        ax = fig.add_subplot(spec5[1, 0])         
        f, t, Pxx = signal.stft(data[:, 0], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        # plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])
             
        ax = fig.add_subplot(spec5[2, 0])   
        plt.plot(data[:, 1] , 'k')
        plt.xlim(0, 6000)  
            
        ax.set_xticks([])
        if len(spt) > 0 and np.count_nonzero(data[:, 1]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2) 
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 1]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        ax = fig.add_subplot(spec5[2, 1])         
        if pl or sl:
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['N', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')
    
    
        ax = fig.add_subplot(spec5[3, 0]) 
        f, t, Pxx = signal.stft(data[:, 1], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        # plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])        
                       
        
        ax = fig.add_subplot(spec5[4, 0]) 
        plt.plot(data[:, 2], 'k') 
        plt.xlim(0, 6000)   
            
        ax.set_xticks([])               
        if len(spt) > 0 and np.count_nonzero(data[:, 2]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2) 
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 2]) > 10:
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)  
                    
        ax = fig.add_subplot(spec5[4, 1])                         
        if pl or sl:    
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], fancybox=True, shadow=True)
            plt.axis('off')        
    
        ax = fig.add_subplot(spec5[5, 0])         
        f, t, Pxx = signal.stft(data[:, 2], fs=100, nperseg=80)
        Pxx = np.abs(Pxx)                       
        # plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
        plt.ylim(0, 40)
        plt.text(1, 1, 'STFT', fontdict=font0)
        plt.ylabel('Hz', fontsize=12)
        ax.set_xticks([])                   
            
        ax = fig.add_subplot(spec5[6, 0])
        x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
                               
        plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=2, label='Earthquake')
        plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=2, label='P_arrival')
        plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=2, label='S_arrival')
        plt.tight_layout()       
        plt.ylim((-0.1, 1.1)) 
        plt.xlim(0, 6000)
        plt.ylabel('Probability', fontsize=12) 
        plt.xlabel('Sample', fontsize=12) 
        plt.yticks(np.arange(0, 1.1, step=0.2))
        axes = plt.gca()
        axes.yaxis.grid(color='lightgray')        
    
        ax = fig.add_subplot(spec5[6, 1])  
        custom_lines = [Line2D([0], [0], linestyle='--', color='g', lw=2),
                        Line2D([0], [0], linestyle='--', color='b', lw=2),
                        Line2D([0], [0], linestyle='--', color='r', lw=2)]
        plt.legend(custom_lines, ['Earthquake', 'P_arrival', 'S_arrival'], fancybox=True, shadow=True)
        plt.axis('off')
            
        font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }
        
        plt.text(1, 0.2, 'EQTransformer', fontdict=font)
        if EQT_VERSION:
            plt.text(2000, 0.05, str(EQT_VERSION), fontdict=font)
            
        plt.xlim(0, 6000)
        fig.tight_layout()
        fig.savefig(os.path.join(save_figs, str(evi).replace(':', '-')+'.png')) 
        plt.close(fig)
        plt.clf()
    

    else:        
        
        ########################################## ploting only in time domain
        fig = plt.figure(constrained_layout=True)
        widths = [1]
        heights = [1.6, 1.6, 1.6, 2.5]
        spec5 = fig.add_gridspec(ncols=1, nrows=4, width_ratios=widths,
                              height_ratios=heights)
        
        ax = fig.add_subplot(spec5[0, 0])         
        plt.plot(data[:, 0], 'k')
        x = np.arange(6000)
        plt.xlim(0, 6000) 
        
        # if platform.system() == 'Windows':  
        #     plt.title(save_figs.split("\\")[-2].split("_")[0]+":"+str(evi))
        # else:
        #     plt.title(save_figs.split("/")[-2].split("_")[0]+":"+str(evi))

        plt.ylabel('Amplitude\nCounts')
                                          
        plt.rcParams["figure.figsize"] = (8,6)
        legend_properties = {'weight':'bold'}  
        
        pl = sl = None        
        if len(spt) > 0 and np.count_nonzero(data[:, 0]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 0]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        if pl or sl:    
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)
                                           
        ax = fig.add_subplot(spec5[1, 0])   
        plt.plot(data[:, 1] , 'k')
        plt.xlim(0, 6000)            
        plt.ylabel('Amplitude\nCounts')            
                  
        if len(spt) > 0 and np.count_nonzero(data[:, 1]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 1]) > 10: 
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
    
        if pl or sl:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['N', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)
                         
        ax = fig.add_subplot(spec5[2, 0]) 
        plt.plot(data[:, 2], 'k') 
        plt.xlim(0, 6000)                    
        plt.ylabel('Amplitude\nCounts')
            
        ax.set_xticks([])
                   
        if len(spt) > 0 and np.count_nonzero(data[:, 2]) > 10:
            ymin, ymax = ax.get_ylim()
            for ipt, pt in enumerate(spt):
                if pt and ipt == 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
                elif pt and ipt > 0:
                    pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                    
        if len(sst) > 0 and np.count_nonzero(data[:, 2]) > 10:
            for ist, st in enumerate(sst): 
                if st and ist == 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
                elif st and ist > 0:
                    sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                    
        if pl or sl:    
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], 
                       loc='center left', bbox_to_anchor=(1, 0.5), 
                       fancybox=True, shadow=True)       
                   
        ax = fig.add_subplot(spec5[3, 0])
        x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
                            
        plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=1.5, label='Earthquake')
        plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=1.5, label='P_arrival')
        plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=1.5, label='S_arrival')
            
        plt.tight_layout()       
        plt.ylim((-0.1, 1.1)) 
        plt.xlim(0, 6000)                                            
        plt.ylabel('Probability') 
        plt.xlabel('Sample')  
        plt.legend(loc='lower center', bbox_to_anchor=(0., 1.17, 1., .102), ncol=3, mode="expand",
                       prop=legend_properties,  borderaxespad=0., fancybox=True, shadow=True)
        plt.yticks(np.arange(0, 1.1, step=0.2))
        axes = plt.gca()
        axes.yaxis.grid(color='lightgray')
            
        font = {'family': 'serif',
                    'color': 'dimgrey',
                    'style': 'italic',
                    'stretch': 'condensed',
                    'weight': 'normal',
                    'size': 12,
                    }
    
        plt.text(6500, 0.5, 'EQTransformer', fontdict=font)
        if EQT_VERSION:
            plt.text(7000, 0.1, str(EQT_VERSION), fontdict=font)
            
        fig.tight_layout()
        fig.savefig(os.path.join(save_figs, str(evi).replace(':', '-')+'.png')) 
        plt.close(fig)
        plt.clf()


def _plot_waveform(
                data,
                args,
                out_dir,
                filename
    ):
    os.makedirs(out_dir, exist_ok=True)
    if args['plot_mode'] == 'time_frequency':
    
        fig = plt.figure(constrained_layout=False)
        widths = [6]
        heights = [1, 1, 1]
        spec5 = fig.add_gridspec(ncols=1, nrows=3, width_ratios=widths,
                              height_ratios=heights, left=0.05, right=0.95, hspace=0.1)
        
        ax = fig.add_subplot(spec5[0, 0])         
        plt.plot(data[:, 0], 'k')
        plt.xlim(0, N_SAMPLES_LMT)
        x = np.arange(N_SAMPLES_LMT)
        ax.set_xticks([])
                     
        plt.rcParams["figure.figsize"] = (10, 10)
        legend_properties = {'weight':'bold'} 

        ax = fig.add_subplot(spec5[1, 0])   
        plt.plot(data[:, 1] , 'k')
        plt.xlim(0, N_SAMPLES_LMT)  
        ax.set_xticks([])
            

        ax = fig.add_subplot(spec5[2, 0]) 
        plt.plot(data[:, 2], 'k') 
        plt.xlim(0, N_SAMPLES_LMT) 
        ax.set_xticks([])

        plt.xlim(0, N_SAMPLES_LMT)

        fig.savefig(os.path.join(out_dir, str(filename).replace(':', '-')+'.png')) 
        plt.close(fig)
        plt.clf()


def _plot_raw(data, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure( figsize=(30, 10))
    fig.add_gridspec(nrows=1, ncols=1, left=0.02, right=0.99)

    plt.plot(data[:, 0], 'k')
    plt.xlim(0, len(data[:, 0]))

    fig.savefig(os.path.join(out_dir, str(filename).replace(':', '-')+'.png')) 
    plt.close(fig)
    plt.clf()


def stream_predictor(
        data, 
        model,
        detection_threshold=0.3,                
        P_threshold=0.1,
        S_threshold=0.1,
        number_of_plots=10,
        plot_mode='time_frequency'
):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%m-%d %H:%M'
    ) 
    
    plot_modes=['time', 'time_frequency']
    args = {
        "detection_threshold": detection_threshold,#
        "P_threshold": P_threshold,#
        "S_threshold": S_threshold,#
        "plot_mode": plot_mode,
        "number_of_plots": number_of_plots
    }

    eqt_logger = logging.getLogger("EQTransformer")
    eqt_logger.info(f"Running EqTransformer")

    pred_start = time.time()
    predD, predP, predS = model.predict(data)
    # print(f"{predD.shape=}")    # (1, 6000, 1) 
    # print(f"{predP.shape=}")    # (1, 6000, 1)
    # print(f"{predS.shape=}")    # (1, 6000, 1)

    plt_n = 0
    out_dir = os.path.join(os.getcwd(), 'stream_outputs')
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for ix in range(len(predD)):      
        args['ix'] = ix
        matches, pick_errors, yh3 =  _picker(args, predD[ix][:, 0], predP[ix][:, 0], predS[ix][:, 0])
        # continue
        #print(f"{matches=}")    # Contains the information for the detected and picked event.

        # _plot_waveform(
        #     data['input'][0],
        #     args,
        #     out_dir,
        #     'waveform', #f"{datetime.now()}"
        # )

        if (len(matches) >= 1) and ((matches[list(matches)[0]][3] or matches[list(matches)[0]][6])):
            # snr = [
            #     _get_snr(data['input'][ix], matches[list(matches)[0]][3], window = 100), 
            #     _get_snr(data['input'][ix], matches[list(matches)[0]][6], window = 100)
            # ]
            # print(snr)

            # for match, match_value in matches.items():
            #     quake_start = match
            #     quake_end = match_value[0]
            #     quake_prob = match_value[1]

            #     p_start = match_value[3]
            #     p_prob = match_value[4]

            #     s_start = match_value[6]
            #     s_prob = match_value[7]

            #     print(f"{quake_start=}, {quake_end=}, {quake_prob=}, {p_start=}, {p_prob=}, {s_start=}, {s_prob=}")

            filename = 'result.png'
            if plt_n < args['number_of_plots']:
                # print('data order: ', data['input'][0][:5, :])
                _plotter_prediction(
                    data['input'][0],
                    args,
                    out_dir,
                    predD[ix][:, 0], 
                    predP[ix][:, 0], 
                    predS[ix][:, 0],
                    filename,
                    matches
                )
                plt_n += 1  
        # else:
        #     _plot_waveform(
        #         data['input'][0],
        #         args,
        #         out_dir,
        #         f"{datetime.now()}"
        #     )

    # print(f"Duration: {time.time() - pred_start}")
    

def read_mseed_data(filepaths):
    offset = 0
    data_enz = np.zeros((6000, 3))

    for filepath in filepaths:
        temp_st = obspy.read(filepath, debub_headers=True)

        try:
            temp_st.merge(fill_value=0) 
        except Exception:
            temp_st =_resampling(temp_st)
            temp_st.merge(fill_value=0)

        channel = temp_st[0].stats.channel[-1]

        # print(temp_st[0].data[:3])

        if channel == 'Z':
            data_enz[:, 2] = temp_st[0].data[offset:offset+6000]
        
        elif channel in ['1', 'E']:
            data_enz[:, 0] = temp_st[0].data[offset:offset+6000]
        
        elif channel in ['2', 'N']:
            data_enz[:, 1] = temp_st[0].data[offset:offset+6000]

    return data_enz
    

def main():
    station_1 = MQTT(Q_MAXSIZE*2, CONFIGS)
    station_1.collect_data()

    ts_data = deque(maxlen=Q_MAXSIZE) 

    data_transform = DataTransform()
    # model = _load_model('EQTransformer/models/EqT_original_model.h5')
    model = _load_model('EQTransformer/models/EqT_model_conservative.h5')

    filename = 'wf-temp'   #f"wf-{datetime.now()}"
    filename_interpolated = 'wf-interpolated'

    """ #####################################################
    # paths = [
    #     '../../downloads_mseeds/B921/PB.B921..EH1__20190901T000000Z__20190902T000000Z.mseed',
    #     '../../downloads_mseeds/B921/PB.B921..EH2__20190901T000000Z__20190902T000000Z.mseed',
    #     '../../downloads_mseeds/B921/PB.B921..EHZ__20190901T000000Z__20190902T000000Z.mseed'
    # ]
    # data_enz = read_mseed_data(paths)

    # trace_e = Trace(
    #     data_enz[:, 0],
    #     header={
    #         'network':'uk', # uk -> unknown
    #         'station':'uk',
    #         'location': 'uk',
    #         'channel':'E',
    #         'sampling_rate':100,
    #         'starttime':UTCDateTime("2024-07-15T00:00:00")
    #     }
    # )
    # trace_n = Trace(
    #     data_enz[:, 1],
    #     header={
    #         'network':'uk', # uk -> unknown
    #         'station':'uk',
    #         'location': 'uk',
    #         'channel':'N',
    #         'sampling_rate':100,
    #         'starttime':UTCDateTime("2024-07-15T00:00:00")
    #     }
    # )
    # trace_z = Trace(
    #     data_enz[:, 2],
    #     header={
    #         'network':'uk', # uk -> unknown
    #         'station':'uk',
    #         'location': 'uk',
    #         'channel':'Z',
    #         'sampling_rate':100,
    #         'starttime':UTCDateTime("2024-07-15T00:00:00")
    #     }
    # )
    # trace_e.detrend('demean')
    # trace_n.detrend('demean')
    # trace_z.detrend('demean')

    # st = Stream(traces=[trace_e, trace_n, trace_z])

    # # Apply filter and taper in the stream
    # st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    # st.taper(max_percentage=0.001, type='cosine', max_length=2)

    # data_transform = DataTransform()
    
    # print(st[0].data[:3])

    ##################################################### """
    
    while True:
        start_time = time.time()
        data = station_1.read()

        if data:
            ts_data.extend(data)
        data = reshape(copy.deepcopy(ts_data))
        # print(f"{data[-10:, 0]}")

        # max_value, min_value = max(data), min(data)
        # print(f"{max_value=}, {min_value=}")
        # print(data.squeeze())

        if True:
            _plot_raw(
                data,
                out_dir=os.getcwd() + '\\stream_outputs\\waveform',
                filename=filename
            )
        # continue


        # create Trace from numpy data
        trace_e = Trace(
            data[:, 0],
            header={
                'network':'uk', # uk -> unknown
                'station':'uk',
                'location': 'uk',
                'channel':'E',
                'sampling_rate':SAMPLE_RATE,
                'starttime':UTCDateTime("2024-07-15T00:00:00")
            }
        )
        # data = reshape(copy.deepcopy(ts_data))
        # create Trace from numpy data
        trace_n = Trace(
            data[:, 0],
            header={
                'network':'uk', # uk -> unknown
                'station':'uk',
                'location': 'uk',
                'channel':'N',
                'sampling_rate':SAMPLE_RATE,
                'starttime':UTCDateTime("2024-07-15T00:00:00")
            }
        )
        # data = reshape(copy.deepcopy(ts_data))
        # create Trace from numpy data
        trace_z = Trace(
            data[:, 0],
            header={
                'network':'uk', # uk -> unknown
                'station':'uk',
                'location': 'uk',
                'channel':'Z',
                'sampling_rate':SAMPLE_RATE,
                'starttime':UTCDateTime("2024-07-15T00:00:00")
            }
        )

        # # Detrend all the traces
        trace_e.detrend('demean')
        trace_n.detrend('demean')
        trace_z.detrend('demean')

        st = Stream(traces=[trace_e, trace_n, trace_z])

        # st = _resampling(st)  # reomved
        st.interpolate(100.033, method="linear")
        if True:
            _plot_raw(
                np.reshape(st[0].data, (st[0].data.shape[0], -1)),
                out_dir=os.getcwd() + '\\stream_outputs\\waveform',
                filename=filename_interpolated
            )


        # # Apply filter and taper in the stream
        st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
        st.taper(max_percentage=0.001, type='cosine', max_length=2)

        data = stream2array([st])

        input_data = data_transform.transform(data)

        # print(f"Input data shape: {input_data['input'].shape}")
        
        stream_predictor(
            input_data,
            model=model,
        )
        print(f"Duratoin {time.time() - start_time}")
        print('-'*50)

        

if __name__ == "__main__":
    main()
    