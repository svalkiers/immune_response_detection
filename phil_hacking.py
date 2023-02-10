import raptcr
from raptcr.constants.hashing import BLOSUM_62
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

TCRDIST_DM = np.maximum(0., np.minimum(4., 4-BLOSUM_62)) # not really a DM


if 1: # compare cdr3 distances for tcrdist vs hashing
    from raptcr.analysis import Repertoire
    import tcrdist
    from tcrdist.tcr_distances import weighted_cdr3_distance
    from raptcr.hashing import Cdr3Hasher
    from raptcr.constants.hashing import DEFAULT_DM
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import linregress

    def get_nonself_nbrs_from_distances(D_in, num_nbrs):
        'not including self in nbr list'
        D = D_in.copy()
        N = D.shape[0]
        inds = np.arange(N)
        D[inds,inds] = 1e6
        nbrs = np.argpartition(D, num_nbrs-1)[:,:num_nbrs]
        nbrs_set = set((i,n) for i,i_nbrs in enumerate(nbrs) for n in i_nbrs)
        return nbrs_set

    ks = [5,10,25,50,100,150]
    ms = [4,8,16,32,64,128]
    #ks = [10]
    #ms = [64]

    # read some epitope-specific paired tcrs
    fname = './data/phil/conga_lit_db.tsv'
    #fname = './data/phil/pregibon_tcrs.tsv'
    df = pd.read_table(fname)
    df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
              inplace=True)

    rep = Repertoire(df)

    N = df.shape[0]
    print('computing TCRdist cdr3 distances:', N)
    start = timer()
    tcrD= np.array([weighted_cdr3_distance(x,y)
                    for x in df.junction_aa for y in df.junction_aa]).reshape((N,N))
    print(f'That took {timer()-start} seconds.')


    plt.figure(figsize=(12,6))
    results = []
    for ii, dmtag in enumerate(['DEFAULT_DM','TCRDIST_DM']):
        dm = locals()[dmtag]

        for m in ms:
            cdr3_hasher = Cdr3Hasher(
                m=m, distance_matrix=dm, trim_left=3, trim_right=2)
            cdr3_hasher.fit()

            vecs = cdr3_hasher.transform(rep)

            rapD = squareform(pdist(vecs))

            rapD_nbrs = get_nonself_nbrs_from_distances(rapD, num_nbrs)
            assert len(rapD_nbrs) == num_nbrs*rapD.shape[0]

            if len(ms) == 1:
                plt.subplot(1,2,ii+1)
                plt.plot(rapD.ravel(), tcrD.ravel(), 'ro', markersize=5, alpha=0.1)
                plt.xlabel('rapTCR dist')
                plt.ylabel('TCRdist')
                plt.title(f'rapTCR distance matrix: {dmtag} m= {m}')

            for k in ks:
                tcrD_nbrs = get_nonself_nbrs_from_distances(tcrD, k)
                rapD_nbrs = get_nonself_nbrs_from_distances(rapD, k)

                #jaccard overlap
                nbr_overlap = len(rapD_nbrs & tcrD_nbrs)/len(rapD_nbrs | tcrD_nbrs)
                combo_nbrs_list =  list(tcrD_nbrs)+list(rapD_nbrs)
                rapD_dists = [rapD[i,j] for i,j in combo_nbrs_list]
                tcrD_dists = [tcrD[i,j] for i,j in combo_nbrs_list]

                reg = linregress(rapD_dists, tcrD_dists)
                results.append(dict(
                    dmtag = dmtag,
                    m = m,
                    k = k,
                    nbr_overlap = nbr_overlap,
                    nbrdist_rvalue = reg.rvalue,
                ))
                print(results[-1])

    results = pd.DataFrame(results)
    outfile = f'cdr3dist_results_N{len(rep)}.tsv'
    results.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

    if len(ms) == 1:
        pngfile = f'cdr3dist_comparison_N{len(rep)}.png'
        #pngfile = f'/home/pbradley/csdat/raptcr/cdr3dist_comparison_N{len(rep)}.png'
        plt.savefig(pngfile)
        print('made:', pngfile)


if 0: # plot setup performance of PynndescentIndex
    vals = np.array([[  10000,   1.113],
                     [  50000,   6.510],
                     [ 100000,  13.075],
                     [ 500000,  74.740],
                     [1500000, 249.026],
                     ])
    plt.figure()
    plt.plot(vals[:,0], vals[:,1])
    plt.scatter(vals[:,0], vals[:,1])
    xn = vals[-1,0]
    for v0,v1 in zip(vals[:-1], vals[1:]):
        x0,y0=v0
        x1,y1=v1
        slope = (y1-y0)/(x1-x0)
        yn = y1 + slope*(xn-x1)
        plt.plot([x1,xn], [y1,yn], ':k')

    plt.xlabel('num TCRs')
    plt.ylabel('seconds')
    plt.title('PynndescentIndex add(...) time')
    plt.show()




if 0: # try some indexing
    from raptcr.analysis import Repertoire
    from raptcr.hashing import Cdr3Hasher
    from raptcr.indexing import FlatIndex, PynndescentIndex

    start = timer()
    if 0:
        fname = '/home/pbradley/csdat/tcrpepmhc/amir/pregibon_test2/tcr_db.tsv'
        df = pd.read_table(fname)
        df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
                  inplace=True)
    if 2:
        fname = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2022-01-22.tsv'
        print('reading:', fname)
        df = pd.read_table(fname)
        df.rename(columns={'vb':'v_call', 'ja':'j_call', 'cdr3b':'junction_aa'},
                  inplace=True)
    print('reading took:', timer() - start)

    rep = Repertoire(df.head(1500000))

    cdr3_hasher = Cdr3Hasher(m=64)
    cdr3_hasher.fit()

    #print('transform:', rep)
    #vecs = cdr3_hasher.transform(rep)


    # initialize index and add data
    print('create index')
    if 0:
        index = FlatIndex(hasher=cdr3_hasher)
        ks = [5,10,15,20,25,100,250]
    else:
        ks = [25]
        index = PynndescentIndex(hasher=cdr3_hasher, k=ks[0])

    print('add the data')
    start = timer()
    index.add(rep)
    print('adding took:', timer() - start)

    query_seqs = list(rep)[:1000]#['CASSKRDRGNGGYTF', "CASSITPGQGTDEQYF"]
    for k in ks:
        print('find',k,'neighbors for m=', len(query_seqs), 'vs N=', len(rep))
        start = timer()
        if len(ks)==1:
            knn_result = index.knn_search()#query_seqs)
        else:
            knn_result = index.knn_search(query_seqs, k=k)
        print('searching took:', timer() - start, k)

    print('DONE')


if 0: # look at blosum distance matrix
    import raptcr.constants.hashing
    import raptcr.constants.base
    import seaborn as sns
    from sklearn.manifold import MDS

    aas = np.array(list(raptcr.constants.base.AALPHABET))
    dm = raptcr.constants.hashing.DEFAULT_DM
    #dm = TCRDIST_DM

    if 0: # show
        plt.figure()
        plt.imshow(dm)
        plt.colorbar()
        plt.xticks(np.arange(20), aas)
        plt.yticks(np.arange(20), aas)
        plt.show()

    if 0: # clustermap the distances
        cm = sns.clustermap(dm)
        inds = cm.dendrogram_row.reordered_ind
        print(aas[inds])
        cm.ax_heatmap.set_yticks(np.arange(20)+0.5)
        cm.ax_heatmap.set_yticklabels(aas[inds])#, rotation=0, fontsize=6);
        inds = cm.dendrogram_col.reordered_ind
        print(aas[inds])
        cm.ax_heatmap.set_xticks(np.arange(20)+0.5)
        cm.ax_heatmap.set_xticklabels(aas[inds])
        plt.show()


    if 0:# check triangle ineq
        for i, a in enumerate(aas):
            print(i,a)
            for j, b in enumerate(aas):
                for k, c in enumerate(aas):
                    if dm[i,k] > dm[i,j] + dm[j,k] +1e-3:
                        print('trifail:',a,b,c, dm[i,k], dm[i,j] + dm[j,k])


    if 2: # try mds
        m = 64
        print('running mds 64')
        mds = MDS(n_components=m, dissimilarity="precomputed", random_state=11,
                  normalized_stress=False)
        vecs = mds.fit_transform(dm)
        print('mds:', mds.stress_)

        dims = np.arange(1,64)
        stressl = []
        for ii in dims:
            #print('running mds', ii)
            mds3 = MDS(n_components=ii, dissimilarity="precomputed", random_state=11,
                       normalized_stress=False)
            vecs3 = mds3.fit_transform(dm)
            print('mds:', ii, mds3.stress_)
            stressl.append(mds3.stress_)


        plt.figure()
        plt.plot(np.sqrt(dims), np.sqrt(stressl))
        plt.scatter(np.sqrt(dims), np.sqrt(stressl))
        locs = plt.xticks()[0]
        plt.xticks(locs, [x**2 for x in locs])
        locs = plt.yticks()[0]
        plt.yticks(locs, [x**2 for x in locs])
        plt.xlabel('embedding dim')
        plt.ylabel('stress')
        plt.show()




