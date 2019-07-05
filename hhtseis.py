"""
hhtseis.py

Hilbert Huang Transform using Emperical Mode Decomposition to extract
Intrinsic Mode Functions, which are monofrequencies. 
Instantaneous attributes are computed for each trace in the supplied segy
and then saved in a seperate segy, resulting in 15 segy attributes.

>python hhtseis.py il2270.sgy --hideplots --plottrace 500
"""
import os.path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segyio
from shutil import copyfile
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy import interpolate
from scipy.signal import savgol_filter

from  pyhht.emd import * 
from pyhht.visualization import plot_imfs

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

def plot_emd_imfs(imfs,ampmean,ampmode,trnum,pdffn,hideplots=True):
    """IMF: Intrinsic Mode Functions"""
    # ioriginal = range(ampmean.size)
    ioriginal = np.arange(ampmean.size)
    fig,ax = plt.subplots(1,8,figsize=(15,10))
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    ax[3].invert_yaxis()
    ax[4].invert_yaxis()
    ax[5].invert_yaxis()
    ax[6].invert_yaxis()
    ax[0].plot(imfs[0,:],ioriginal,c='r',label='Original')
    ax[0].legend()
    ax[1].plot(ampmean,ioriginal,c='r',label='Mean')
    ax[1].legend()
    ax[2].plot(ampmode,ioriginal,c='r',label='Mode')
    ax[2].legend()
    ax[3].plot(imfs[0,:],ioriginal,c='r',label='IMF1')
    ax[3].legend()
    ax[4].plot(imfs[1,:],ioriginal,c='r',label='IMF1')
    ax[4].legend()
    ax[5].plot(imfs[2,:],ioriginal,c='r',label='IMF2')
    ax[5].legend()
    ax[6].plot(imfs[3,:],ioriginal,c='r',label='IMF3')
    ax[6].legend()
    ax[7].plot(imfs[-1,:],ioriginal,c='r',label='Trend')
    ax[7].legend()
    fig.suptitle(f'Decomposition at trace# {trnum}')
    fig.tight_layout()
    pdffn.savefig()
    if not hideplots:
        plt.show()
    plt.close()

    return fig

def plot_inst_attributes(imfn,env,iph,ifreq,imfnum,trnum,pdffn,hideplots=True):
    """Plot instantaneous attributes per imf"""
    # ioriginal = range(ampmean.size)
    ioriginal = np.arange(env.size)
    fig,ax = plt.subplots(1,4,figsize=(15,10))
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    ax[3].invert_yaxis()
    ax[0].plot(imfn,ioriginal,c='r',label='IMF%-d'%imfnum)
    ax[0].legend()
    ax[1].plot(env,ioriginal,c='r',label='Envelope')
    ax[1].legend()
    ax[2].plot(iph,ioriginal,c='r',label='InstPhase')
    ax[2].legend()
    ax[3].plot(ifreq,ioriginal,c='r',label='InstFreq')
    ax[3].legend()
    fig.suptitle(f'Instantaneous Attributes of IFM{imfnum} at trace# {trnum}')
    fig.tight_layout()
    pdffn.savefig()
    if not hideplots:
        plt.show()
    plt.close()


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



def getcommandline():
    """Main."""
    parser = argparse.ArgumentParser(description='Hilbert Huang Transform: Empirical Mode Decomposition generating Intrinsic Mode Functions
')
    parser.add_argument('segyfile', help='segy file name')
    # parser.add_argument('--tracerange',type=int,nargs=2,default=[0,-1],help='Start and end trace #s. default full range')
    parser.add_argument('--numofimf',type=int,default=0,help='# of IMF to calculate. default= program decides, i.e. variable')
    parser.add_argument('--ifreqsmoothwlen',type=int,default=21,help='smooth ifreq window length. default = 21')
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

    outfn = fname 
    if cmdl.outdir:
        ampmeanfname = os.path.join(cmdl.outdir, outfn) + "_ampmean.sgy"
        ampmodefname = os.path.join(cmdl.outdir, outfn) + "_ampmode.sgy"
        imf0fname = os.path.join(cmdl.outdir, outfn) + "_imf0.sgy"
        imf0envfname = os.path.join(cmdl.outdir, outfn) + "_imf0env.sgy"
        imf0ifreqfname = os.path.join(cmdl.outdir, outfn) + "_imf0ifreq.sgy"
        imf1fname = os.path.join(cmdl.outdir, outfn) + "_imf1.sgy"
        imf1envfname = os.path.join(cmdl.outdir, outfn) + "_imf1env.sgy"
        imf1ifreqfname = os.path.join(cmdl.outdir, outfn) + "_imf1ifreq.sgy"
        imf2fname = os.path.join(cmdl.outdir, outfn) + "_imf2.sgy"
        imf2envfname = os.path.join(cmdl.outdir, outfn) + "_imf2env.sgy"
        imf2ifreqfname = os.path.join(cmdl.outdir, outfn) + "_imf2ifreq.sgy"
        imf3fname = os.path.join(cmdl.outdir, outfn) + "_imf3.sgy"
        imf3envfname = os.path.join(cmdl.outdir, outfn) + "_imf3env.sgy"
        imf3ifreqfname = os.path.join(cmdl.outdir, outfn) + "_imf3ifreq.sgy"
        trendfname = os.path.join(cmdl.outdir, outfn) + "_trend.sgy"
        emdpdf = os.path.join(cmdl.outdir, outfn) + "_emd.pdf"
    else:
        ampmeanfname = os.path.join(dirsplit, outfn) + "_ampmean.sgy"
        ampmodefname = os.path.join(dirsplit, outfn) + "_ampmode.sgy"
        imf0fname = os.path.join(dirsplit, outfn) + "_imf0.sgy"
        imf0envfname = os.path.join(dirsplit, outfn) + "_imf0env.sgy"
        imf0ifreqfname = os.path.join(dirsplit, outfn) + "_imf0ifreq.sgy"
        imf1fname = os.path.join(dirsplit, outfn) + "_imf1.sgy"
        imf1envfname = os.path.join(dirsplit, outfn) + "_imf1env.sgy"
        imf1ifreqfname = os.path.join(dirsplit, outfn) + "_imf1ifreq.sgy"
        imf2fname = os.path.join(dirsplit, outfn) + "_imf2.sgy"
        imf2envfname = os.path.join(dirsplit, outfn) + "_imf2env.sgy"
        imf2ifreqfname = os.path.join(dirsplit, outfn) + "_imf2ifreq.sgy"
        imf3fname = os.path.join(dirsplit, outfn) + "_imf3.sgy"
        imf3envfname = os.path.join(dirsplit, outfn) + "_imf3env.sgy"
        imf3ifreqfname = os.path.join(dirsplit, outfn) + "_imf3ifreq.sgy"
        trendfname = os.path.join(dirsplit, outfn) + "_trend.sgy"
        emdpdf = os.path.join(dirsplit, outfn) + "_emd.pdf"
    print('Copying files, please wait ........')
    start_copy = datetime.now()
    copyfile(cmdl.segyfile, ampmeanfname)
    copyfile(cmdl.segyfile, ampmodefname)
    copyfile(cmdl.segyfile, imf0fname)
    copyfile(cmdl.segyfile, imf0envfname)
    copyfile(cmdl.segyfile, imf0ifreqfname)
    copyfile(cmdl.segyfile, imf1fname)
    copyfile(cmdl.segyfile, imf1envfname)
    copyfile(cmdl.segyfile, imf1ifreqfname)
    copyfile(cmdl.segyfile, imf2fname)
    copyfile(cmdl.segyfile, imf2envfname)
    copyfile(cmdl.segyfile, imf2ifreqfname)
    copyfile(cmdl.segyfile, imf3fname)
    copyfile(cmdl.segyfile, imf3envfname)
    copyfile(cmdl.segyfile, imf3ifreqfname)
    copyfile(cmdl.segyfile, trendfname)
    end_copy = datetime.now()
    print('Duration of copying: {}'.format(end_copy - start_copy))


    start_processing = datetime.now()
    with segyio.open(cmdl.segyfile,'r') as sfn:
        # dt = sfn.bin[sfn.bin.keys()[5]] /1000000.0
        fs = 250 # Need to check that 
        trlen = len(sfn.trace[10])
        with segyio.open(ampmeanfname,'r+') as meanfn,segyio.open(ampmodefname,'r+') as modefn,\
            segyio.open(imf0fname,'r+') as imf0fn, segyio.open(imf0envfname,'r+') as imf0envfn, segyio.open(imf0ifreqfname,'r+') as imf0ifreqfn,\
            segyio.open(imf1fname,'r+') as imf1fn, segyio.open(imf1envfname,'r+') as imf1envfn, segyio.open(imf1ifreqfname,'r+') as imf1ifreqfn,\
            segyio.open(imf2fname,'r+') as imf2fn,segyio.open(imf2envfname,'r+') as imf2envfn, segyio.open(imf2ifreqfname,'r+') as imf2ifreqfn,\
            segyio.open(imf3fname,'r+') as imf3fn,segyio.open(imf3envfname,'r+') as imf3envfn, segyio.open(imf3ifreqfname,'r+') as imf3ifreqfn,\
            segyio.open(trendfname,'r+') as trndfn:
            with PdfPages(emdpdf) as pdf:
                trnum = cmdl.tracerange[0]
                # for tr in sfn.trace[cmdl.tracerange[0]:cmdl.tracerange[1]]:
                for tr in sfn.trace:
                    if trnum % cmdl.plottrace == 0:
                        print(f'Processing: {trnum}')
                    imfsout,amean,amode = emd_trace(tr)
                    meanfn.trace[trnum] = amean
                    modefn.trace[trnum] = amode
                    trenv,triph,trifreq = inst_attributes(tr,freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
                    numimfs = imfsout.shape[0]
                    nimfs = 0
                    if nimfs < numimfs :
                        imf0fn.trace[trnum] = imfsout[nimfs,:]
                        imf0env,iph,imf0ifreq = inst_attributes(imfsout[nimfs,:],freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
                        imf0envfn.trace[trnum] = imf0env
                        imf0ifreqfn.trace[trnum] = imf0ifreq
                    else:
                        print(f'No data found for trace# {trnum}, output zeros')
                        imf0fn.trace[trnum] = np.zeros(trlen)
                        imf0envfn.trace[trnum] =  np.zeros(trlen)
                        imf0ifreqfn.trace[trnum] =  np.zeros(trlen)
                    nimfs += 1  # nimfs = 1
                    if nimfs < numimfs:
                        imf1fn.trace[trnum] = imfsout[nimfs,:]
                        imf1env,iph,imf1ifreq = inst_attributes(imfsout[nimfs,:],freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
                        imf1envfn.trace[trnum] = imf1env
                        imf1ifreqfn.trace[trnum] = imf1ifreq
                    else:
                        imf1fn.trace[trnum] = np.zeros(trlen)
                        imf1envfn.trace[trnum] =  np.zeros(trlen)
                        imf1ifreqfn.trace[trnum] =  np.zeros(trlen)
                    nimfs += 1  # nimfs = 2
                    if nimfs < numimfs:
                        imf2fn.trace[trnum] = imfsout[nimfs,:]
                        imf2env,iph,imf2ifreq = inst_attributes(imfsout[nimfs,:],freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
                        imf2envfn.trace[trnum] = imf2env
                        imf2ifreqfn.trace[trnum] = imf2ifreq
                    else:
                        imf2fn.trace[trnum] = np.zeros(trlen)
                        imf2envfn.trace[trnum] =  np.zeros(trlen)
                        imf2ifreq.trace[trnum] =  np.zeros(trlen)
                    nimfs += 1  # nimfs = 3
                    if nimfs < numimfs:
                        imf3fn.trace[trnum] = imfsout[nimfs,:]
                        imf3env,iph,imf3ifreq = inst_attributes(imfsout[nimfs,:],freq_sample=fs,smoothwlen=cmdl.ifreqsmoothwlen)
                        imf3envfn.trace[trnum] = imf3env
                        imf3ifreqfn.trace[trnum] = imf3ifreq
                    else:
                        imf3fn.trace[trnum] = np.zeros(trlen)
                        imf3envfn.trace[trnum] =  np.zeros(trlen)
                        imf3ifreqfn.trace[trnum] =  np.zeros(trlen)
                    trndfn.trace[trnum] = imfsout[-1,:]

                    if trnum % cmdl.plottrace == 0:
                        plot_emd_imfs(imfsout,amean,amode,trnum,pdf,cmdl.hideplots)
                        plot_inst_attributes(imfsout[0,:],imf0env,iph,imf0ifreq,0,trnum,pdf,hideplots=cmdl.hideplots)
                        plot_inst_attributes(imfsout[1,:],imf1env,iph,imf1ifreq,1,trnum,pdf,hideplots=cmdl.hideplots)
                        plot_inst_attributes(imfsout[2,:],imf2env,iph,imf2ifreq,2,trnum,pdf,hideplots=cmdl.hideplots)
                        plot_inst_attributes(imfsout[3,:],imf3env,iph,imf3ifreq,3,trnum,pdf,hideplots=cmdl.hideplots)
                    trnum +=1


    end_processing = datetime.now()
    print(f'Duration of processing: {end_processing - start_processing}')



if __name__ == '__main__':
    main()
