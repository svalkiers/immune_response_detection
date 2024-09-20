import pandas as pd
import numpy as np

def get_metrics(df, target_col="responding"):
    fp = 0
    tp = 0
    fpr = []
    tpr = []
    for i in df[target_col]:
        if i == False:
            fp += 1
        else:
            tp += 1
        fpr.append(fp)
        tpr.append(tp)
    N = df[target_col].value_counts()[False]
    P = df[target_col].value_counts()[True]
    fpr = [i/N for i in fpr]
    tpr = [i/P for i in tpr]
    return fpr, tpr

def annotate_results(df, responding, target_col):
    if "v_call" in responding.columns:
        df = df.merge(responding, on=["junction_aa","v_call"], how="left")
    else:
        if "v_gene" not in df.columns:
            df["v_gene"] = df.v_call.apply(lambda x: x.split("*")[0])
            df = df.merge(responding, on=["junction_aa","v_gene"], how="left")
    df[target_col] = df[target_col].fillna(False)
    # df["responding"] = df.responding.map({"red":True,"grey":False})
    return df

def identify_neighbors(n, d=1):
    if n > d:
        return False
    else:
        return True

def get_distance_from_responding(df):
    lds = []
    for i in range(len(df)):
        seq = df.iloc[i]
        cdr3 = seq["junction_aa"]
        v = seq["v_gene"]
        vsub = responding[responding.v_gene==v]
        try:
            lds.append(min([levenshtein_distance(cdr3,j) for j in vsub.junction_aa]))
        except ValueError:
            lds.append(None)
    df['ld'] = lds
    df["hit"] = df.ld.apply(lambda x: identify_neighbors(x))
    return df