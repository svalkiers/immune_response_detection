# apologies: old code, quickly ported...
import tcrdist
from tcrdist.all_genes import all_genes # for recognized genes
from tcrdist.translation import get_translation
from tcrdist.tcr_sampler import get_j_cdr3_nucseq
import pandas as pd

organism = 'mouse'

expected_bad_vgenes = set( [ 'TCRBV22-01' ] )


def convert_to_imgt_v_gene( vg ):
    assert vg.startswith('TCRBV')
    assert len(vg) == 10 and vg[-3] == '-'
    dashnum = int(vg[8:10])
    new_vg_yesdash = 'TRBV{}-{}*01'.format( int(vg[5:7]), dashnum )
    if dashnum>1:
        new_vg_nodash = None
    else:
        new_vg_nodash = 'TRBV{}*01'.format( int(vg[5:7]) )
    return new_vg_nodash, new_vg_yesdash

def convert_to_imgt_j_gene( jg ):
    assert jg.startswith('TCRBJ')
    assert len(jg) == 10 and jg[-3] == '-'
    return 'TRBJ{}-{}*01'.format( int(jg[5:7]), int(jg[8:10] ) )


def parse_tcr( tcr ): # tcr was made by read_adaptive_tsvfile
    ''' Returns None on failure
    otherwise returns   ( vfam, vg, jg, cdr3, cdr3_nucseq )
    '''

    vfam, vgene, cdr3, rearrangement, jgene = tcr

    if len(cdr3)<=5:
        print('short cdr3:',cdr3)
        return None
    if not vgene or not jgene: # empty: unresolved probably
        print('empty genes {} {} {}'.format(vfam,vgene,jgene))
        return None
    if not vgene.startswith('TCRBV'):
        print('funny vgene: ({})'.format(vgene))
        return None
    if vgene in expected_bad_vgenes:
        print('bad vgene: ({})'.format(vgene))
        return None
    vg1, vg2 = convert_to_imgt_v_gene( vgene )
    jg = convert_to_imgt_j_gene( jgene )

    if vg2 in all_genes[organism]:
        vg = vg2
    elif vg1 in all_genes[organism]:
        vg = vg1
    else:
        print('bad vg:', vgene, vg1, vg2)
        return None

    # figure out the cdr3 nucseq
    cdr3_nucseq = None
    jg_cdr3_nucseq = get_j_cdr3_nucseq( 'mouse',jg ).upper() # added .upper()

    for offset in range(3):
        protseq = get_translation( rearrangement, '+{}'.format(offset+1) )
        #print offset, cdr3, protseq, rearrangement
        for ctrim in range(1,4):
            if cdr3[:-ctrim] in protseq:
                start = offset + 3*(protseq.index(cdr3[:-ctrim]))
                length = 3*(len(cdr3)-ctrim)
                cdr3_nucseq = rearrangement[ start : start + length ]
                if ctrim:
                    cdr3_nucseq += jg_cdr3_nucseq[-3*ctrim:]
                #print cdr3, get_translation( cdr3_nucseq, '+1' )[0], jg_cdr3_nucseq, rearrangement
                if cdr3 != get_translation( cdr3_nucseq, '+1' ):
                    print('translation fail:', cdr3,
                          get_translation( cdr3_nucseq, '+1' ), \
                          jg_cdr3_nucseq, rearrangement)
                    cdr3_nucseq = ''
                break
        if cdr3_nucseq=='': # failure signal
            cdr3_nucseq = None
            break

    if cdr3_nucseq is None:
        print('parse cdr3_nucseq failed:',tcr)
        return None

    return ( vfam, vg, jg, cdr3, cdr3_nucseq )


def parse_file(fname):
    print('reading:', fname)
    df = pd.read_table(fname)

    mask = ~(df.v_gene.isna() | df.j_gene.isna() | df.amino_acid.isna() |
             df.rearrangement.isna())
    df = df[mask]

    tcrs = []
    for l in df.itertuples():

        frame = l.frame_type
        if frame != 'In':
            assert frame in ['Out','Stop']
            continue
        templates = l.templates

        # kludgy old for backwards-compat
        tcr = ( l.v_family, l.v_gene, l.amino_acid, l.rearrangement, l.j_gene)

        newtcr = parse_tcr(tcr)

        if newtcr is None:
            print('parse fail:', tcr)
        else:
            tcrs.append(newtcr)
    return tcrs

fname = '/home/pbradley/csdat/hill/1901T-B6_45_1-donor-spleen-A.tsv'

tcrs = parse_file(fname)
