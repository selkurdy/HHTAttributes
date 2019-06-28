"""
mlhht 
Machine learning hilbert huang transform
Takes one segy and generates 21 frequency envelope and phae attributes that are saved
to a npy binary file and csv file.

>python mlhht.py il2270.sgy
"""
import os.path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segyio
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy import interpolate
from scipy.signal import savgol_filter
import itertools

from  pyhht.emd import * 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
from scipy import interpolate

from pylab import rcParams
import segyio



def emd_trace(trc,nimfs=0):
    """Trace EMD : Emperical Mode Decomposition"""
    ioriginal = range(trc.size)

    decomposer = EMD(trc,n_imfs=nimfs)
    imfs = decomposer.decompose()
    # plot_imfs(tr100[0,:], imfs, ioriginal)    
    mean_amp = decomposer.mean_and_amplitude(trc)
    # mean_amp[0] is mean amplitude
    # amp[3] is mode or envelop
    return imfs,mean_amp[0],mean_amp[3] 


def smooth(a, wlen=11, mode='valid') :
    if wlen % 2 == 0:  #window has to be odd
        wlen +=1
    asmth = savgol_filter(a, wlen, 3) # window size 51, polynomial order 3
    return asmth


def inst_attributes(tr,freq_sample=500,smoothwlen=11):
    """Compute instantaneous attributes"""
    analytic_signal = hilbert(tr)
    aenv = np.abs(analytic_signal)
    iph = np.unwrap(np.angle(analytic_signal))
    ifreq = np.abs((np.diff(iph,prepend=0) / (2.0*np.pi) * freq_sample))
    ifreqsmth = smooth(ifreq,smoothwlen,mode='same')

    return aenv,iph,ifreqsmth

def zero_segy(fname):
    with segyio.open(fname,'r+',ignore_geometry= True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            srcp.trace[trnum] = tr * 0



def get_samplerate(fname):
    with segyio.open(fname,'r',ignore_geometry= True) as srcp:
        hdrdict = dict(enumerate(srcp.header[1].items()))
    return hdrdict[39][1]/1000




def getcommandline():
    """Main."""
    parser = argparse.ArgumentParser(description='Hilbert Huang Transform HHT or Empirical Mode Decomposition EMD')
    parser.add_argument('segyfile', help='segy file name')
    parser.add_argument('--startendslice',type=int,nargs=2,default=[500,1500],help='Start end slice. default= 500 to 1500 ms')
    parser.add_argument('--numofimf',type=int,default=0,help='# of IMF to calculate. default= program decides, i.e. variable')

    parser.add_argument('--ifreqsmoothwlen',type=int,default=21,help='smooth ifreq window length. default = 21')
    parser.add_argument('--colnames',action='store_true',default=False,help='List column names of data file.default=False')

    parser.add_argument('--plottrace',type=int,default=50000,
        help='plot increment. default=50000')
    parser.add_argument('--outdir',help='output directory,default= same dir as input')
    parser.add_argument('--hideplots',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    result = parser.parse_args()
    return result



def main():
    """main"""

    import warnings
    warnings.filterwarnings("ignore")

    cmdl = getcommandline()
    dirsplit,fextsplit = os.path.split(cmdl.segyfile)
    fname, fextn = os.path.splitext(fextsplit)

    dt = get_samplerate(cmdl.segyfile)
    sstart = int(cmdl.startendslice[0] // dt)
    # sample start
    send = int(cmdl.startendslice[1] // dt)
    # sample end

    outfn = fname 
    if cmdl.outdir:
        npyfname = os.path.join(cmdl.outdir, outfn) + "_%d_%d.npy" %(sstart,send)
        dffname = os.path.join(cmdl.outdir, outfn) + "_%d_%d.csv" %(sstart,send)
        
    else:
        npyfname = os.path.join(dirsplit, outfn) + ".npy"
        dffname = os.path.join(dirsplit, outfn) + ".csv"
    start_processing = datetime.now()
    with segyio.open(cmdl.segyfile,'r') as sfn:
        fs = 250 # Need to check that 
        print(f'#of traces in file {len(sfn.trace)}')
        # for tra in sfn.trace[cmdl.tracerange[0]:cmdl.tracerange[1]]:
        for trnum,tra in enumerate(sfn.trace):
            tr = tra[sstart:send]
            if trnum % cmdl.plottrace == 0:
                print(f'Processing Trace# : {trnum}')
            trnaray = np.full(shape=tr.size,fill_value=trnum,dtype=np.int)
            imfsout,amean,amode = emd_trace(tr)
            # imfsT = imfsout.T
            trenv,triph,trifreq = inst_attributes(tr,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
            imf0 = imfsout[0,:].T
            imf0env,imf0iph,imf0ifreq = inst_attributes(imf0,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
            imf1 = imfsout[1,:].T
            imf1env,imf1iph,imf1ifreq = inst_attributes(imf1,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
            imf2 = imfsout[2,:].T
            imf2env,imf2iph,imf2ifreq = inst_attributes(imf2,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
            imf3 = imfsout[3,:].T
            imf3env,imf3iph,imf3ifreq = inst_attributes(imf3,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)

            
            # print(f'imf0 shape {imf0.shape}')
            trall = np.vstack((trnaray,amean,amode,trenv,triph,trifreq,imf0,imf0env,imf0iph,imf0ifreq,\
                            imf1,imf1env,imf1iph,imf1ifreq,imf2,imf2env,imf2iph,imf2ifreq,\
                            imf3,imf3env,imf3iph,imf3ifreq))
            if trnum == 0:
                alldata = trall
            else:
                alldata = np.hstack((alldata,trall))
            # print(f' alldata hstack {alldata.shape}')
        alldata = alldata.T
        print(f' alldata after transpose {alldata.shape}')
        np.save(npyfname,alldata)
        print(f'Successfully saved {npyfname}')

    datacols = ['TRACENUM','MEANENV','MODEAMP','ENVELOP','INSTPHASE','INSTFREQ',\
        'IMF0AMPLITUDE','IMF0ENVELOP','IMF0INSTPHASE','IMF0INSTFREQ',\
            'IMF1AMPLITUDE','IMF1ENVELOP','IMF1INSTPHASE','IMF1INSTFREQ',\
                'IMF2AMPLITUDE','IMF2ENVELOP','IMF2INSTPHASE','IMF2INSTFREQ',\
                    'IMF3AMPLITUDE','IMF3ENVELOP','IMF3INSTPHASE','IMF3INSTFREQ']
    alldf = pd.DataFrame(alldata,columns=datacols)
    # print(alldf.head())
    print(alldf.describe().T)
    alldf.to_csv(dffname,index=False)
    print(f'Sucessfully generated {dffname}')
    if cmdl.colnames:
        for i,name in enumerate(datacols):
            print(f'{i:5}  {name}') 
    print(f'alldata shape: {alldata.shape}')
    # print(len(datacols))


    end_processing = datetime.now()
    print(f'Duration of processing: {end_processing - start_processing}')




if __name__ == '__main__':
    main()
