"""
clustering2segy.py
take a segy file and a csv file generated from mlhht.py :
    ~copy segy and zero amplitudes to a new file
    ~read csv file and dimensionality reduce using UMAP - takes a long time
    ~use Kmeans to cluster data to 5 clusters
    ~output results to a csv as one column called CLUSTER
    ~write data to the zeroed segy

    >python clustering2segy.py il2270.csv il2270.sgy --clusterdatacsv il2270_5clstr.csv
    >python clustering2segy.py il2270.csv il2270.sgy 
"""

import os.path
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segyio
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import itertools
from sklearn.cluster import KMeans
from shutil import copyfile

try:
    import umap
except ImportError:
    print('***Warning: umap is not installed')


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
    parser = argparse.ArgumentParser(description='Dimensionality Reduction and clustering written back to segy')
    parser.add_argument('datacsvfile',help='Previously saved data array from mlhht.py ')
    parser.add_argument('segyfile', help='segy file name to fill with clusters')
    # parser.add_argument('--xhdr',type=int,default=73,help='xcoord header.default=73')
    # parser.add_argument('--yhdr',type=int,default=77,help='ycoord header. default=77')
    # parser.add_argument('--xyscalerhdr',type=int,default=71,help='hdr of xy scaler to divide by.default=71')
    # parser.add_argument('--tracerange',type=int,nargs=2,default=[0,-1],help='Start and end trace #s. default full range')
    parser.add_argument('--startendslice',type=int,nargs=2,default=[500,1500],help='Start end slice. default= 500 to 1500 ms')
    parser.add_argument('--scalekind',choices=['standard','quniform','qnormal'],default='standard',
        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')

    parser.add_argument('--nneighbors',type=int,default=20,help='Nearest neighbors. default=20')
    parser.add_argument('--mindistance',type=float,default=0.3,help='Min distantce for clustering. default=0.3')
    parser.add_argument('--ncomponents',type=int,default=3,help='Projection axes. default=3')
    parser.add_argument('--nclusters',type=int,default=5,
        help='Clustering after dimensionality reduction by umap.default=5')
    parser.add_argument('--clusterdatacsv',default=None,help='csv with cluster column. Will not UMAP and cluster. Fill zeroed segy only ')
    # parser.add_argument('--plottrace',type=int,default=50000,
    #     help='plot increment. default=50000')
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
    dirsplit,fextsplit = os.path.split(cmdl.datacsvfile)
    fname, fextn = os.path.splitext(fextsplit)

    dt = get_samplerate(cmdl.segyfile)
    sstart = int(cmdl.startendslice[0] // dt)
    # sample start
    send = int(cmdl.startendslice[1] // dt)
    # sample end
    print(f'Seismic Sample Rate: {dt}, start sample#: {sstart}, end sample#: {send}')


    outfn = fname 
    if cmdl.outdir:
        pdfcl = os.path.join(cmdl.outdir,fname) + "_umapnc%d.pdf" % (cmdl.ncomponents)
        npyfname = os.path.join(cmdl.outdir, outfn) + "_umaplbl.npy"
        clusteredfname = os.path.join(cmdl.outdir, outfn) + "_%dclstr.csv" % cmdl.nclusters
        outfsegy = os.path.join(cmdl.outdir,outfn) +"_%dclstr.sgy" % (cmdl.nclusters)
        
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_umapnc%d.pdf" % (cmdl.ncomponents)
        npyfname = os.path.join(dirsplit, outfn) + "_umaplbl.npy"
        clusteredfname = os.path.join(dirsplit, outfn) + "_%dclstr.csv" % cmdl.nclusters
        outfsegy = os.path.join(dirsplit,outfn) +"_%dclstr.sgy" % (cmdl.nclusters)

    if not cmdl.clusterdatacsv:
        # alldata = np.load(cmdl.datafile)
        alldatadf = pd.read_csv(cmdl.datacsvfile)
        print(alldatadf.shape)
        # alldata = alldatadf[:,1:].values

        if cmdl.scalekind == 'standard':
            alldatasc = StandardScaler().fit_transform(alldatadf.iloc[:,1:].values)
        elif cmdl.scalekind == 'quniform':
            alldatasc = QuantileTransformer(output_distribution='uniform').fit_transform(alldatadf.iloc[:,1:].values)
        else:
            alldatasc = QuantileTransformer(output_distribution='normal').fit_transform(alldatadf.iloc[:,1:].values)

        # alldatascdf = pd.DataFrame(alldatasc,columns=cols)

        clustering = umap.UMAP(n_neighbors=cmdl.nneighbors, min_dist=cmdl.mindistance, n_components=cmdl.ncomponents)

        start_time = datetime.now()

        print('Start UMAP Clustering')
        umap_features = clustering.fit_transform(alldatasc)
        print(f'umap features shape {umap_features.shape}')
        end_time = datetime.now()
        print('UMAP dimensionality Reduction Duration: {}'.format(end_time - start_time))

        clustering = KMeans(n_clusters=cmdl.nclusters,
            n_init=5,
            max_iter=300,
            tol=1e-04,
            random_state=1)
        ylabels = clustering.fit_predict(umap_features)
        nlabels = np.unique(ylabels)
        print('nlabels',nlabels)

        print(f'Labelled  shape: {ylabels.shape}  umap features shape: {umap_features.shape}')
        # label_data = np.vstack((umap_features,ylabels))
        # np.save(npyfname)

        alldatadf['CLUSTER'] = ylabels
        alldatadf.to_csv(clusteredfname,index=False)
        print(f'Successfully generated {clusteredfname}')

        fig, ax = plt.subplots(figsize=(8,6))
        nclst = [i for i in range(cmdl.ncomponents)]
        pltvar = itertools.combinations(nclst,2)
        pltvarlst = list(pltvar)
        for i in range(len(pltvarlst)):
            ftr0 = pltvarlst[i][0]
            ftr1 = pltvarlst[i][1]
            print('umap feature #: {}, umap feature #: {}'.format(ftr0,ftr1))
            plt.scatter(umap_features[:,ftr0],umap_features[:,ftr1],c=ylabels,s=2,alpha=.2)
            plt.show()
            plt.close()

    else:
        alldatadf = pd.read_csv(cmdl.clusterdatacsv)

    print('Copying file, please wait ........')
    start_copy = datetime.now()
    copyfile(cmdl.segyfile, outfsegy)
    end_copy = datetime.now()
    print('Duration of copying: {}'.format(end_copy - start_copy))


    print('Zeroing segy file, please wait ........')
    start_zero = datetime.now()
    zero_segy(outfsegy)
    end_zero = datetime.now()
    print('Duration of zeroing: {}'.format(end_zero - start_zero))

    with segyio.open( outfsegy, "r+" ) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            tr[sstart: send] = alldatadf[alldatadf.TRACENUM==trnum]['CLUSTER'].values
            srcp.trace[trnum] = tr
    print('Successfully generated {}'.format(outfsegy))


if __name__ == '__main__':
    main()
