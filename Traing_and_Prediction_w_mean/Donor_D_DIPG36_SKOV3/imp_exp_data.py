import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
data_path = Path.cwd().parent.parent / 'data/SKOV3_DIPG36_Her2'
def new_data(df):
    mean,sd = df.iloc[:,0],df.iloc[:,1]#/np.sqrt(3)
    a = np.random.normal(mean,sd,(10,5))
    b = [np.random.choice(a[:,i][(a[:,i]>=mean[i]-sd[i])&(a[:,i]<=mean[i]+sd[i])]) for i in range(len(df))]
    return (mean.values).tolist(),(sd.values).tolist() #b #(mean.values).tolist()


def R_L_data(NK_cell, Tumor_cell,limts,CAR_Gen=None):
    data_R_type = {}
    data_L_type = {}
    keys = ['CAR','LFA1','KIR']
    for i,key in enumerate(keys):
        if key == 'CAR':
            key = key if CAR_Gen == None else CAR_Gen 
        df = pd.read_excel(f'{data_path}/{NK_cell}.xlsx',sheet_name=f'{NK_cell}_{key}')
        data_R_type[f'{NK_cell}_{key}'] = adding_prob_dens(df,Cell_type='NK_cell',MFI_range=limts[0][i])
    #Plots(data_R_type,cell_type = NK_cell)
    keys = (pd.ExcelFile(f'{data_path}/{Tumor_cell}.xlsx')).sheet_names
    for i,key in enumerate(keys):
        df = pd.read_excel(f'{data_path}/{Tumor_cell}.xlsx',sheet_name=key)
        data_L_type[key] = adding_prob_dens(df,MFI_range=limts[1][i],Cell_type='Tumor_cell')
    #Plots(data_L_type,cell_type = Tumor_cell)
    return data_R_type, data_L_type

def coefficients(equation):
    numbers = re.findall(r'-?\d+\.?\d*', equation)
    coefficients = [float(num) for num in numbers]
    return coefficients
def Plots(cell,cell_type = None,frac=None):
    n_cols = len(cell)
    plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})
    fig, ax = plt.subplots(3, n_cols, figsize=(6 * n_cols, 8), constrained_layout=True)
    for j, key in enumerate(cell.keys()):
        nmbr, prob, bins, df,thrs = cell[key]
        title = f'{key}'
        ax[0, j].plot(df.iloc[:, 0], df.iloc[:, 1], 'd', markersize=1)
        ax[0, j].set_title(title, fontsize=24)
        ax[1, j].plot(df.iloc[:, 2], df.iloc[:, 1], 'd', markersize=1)
        bar_colors = ['red'] + ['green'] * (len(cell[key][2]) - 1) if thrs!=None and j in [0, 2] else 'green'
        ax[2, j].bar(bins[:-1], prob, width=np.diff(bins), align='edge',
             alpha=0.5, color=bar_colors, edgecolor='black')
        for i in [1, 2]:  # Only bottom two rows
            ax[i, j].tick_params(axis='x', rotation=45)
    for axs in ax.flat:
        for spine in axs.spines.values():
            spine.set_linewidth(1.5)
    plt.savefig(f'{cell_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def prob_density_R(df, MFI_range):
    equation = df.apply(lambda col: col.dropna()[col.dropna().astype(str).str.contains('molecules|positivity threshold', case=False)] if col.dtype == 'object' else pd.Series(), axis=0).stack().values
    coeff = coefficients(equation[0])
    df = df.loc[(df.iloc[:,0] > MFI_range[0]) & (df.iloc[:,0] <= MFI_range[1])& (df.iloc[:,1] > 0)].reset_index(drop=True)
    df = df.iloc[:,:2]
    thrs = None
    if len(equation) > 1:
        thrs = coefficients(equation[1])[0]
    df['nmbr_mcule'] = coeff[0]**(coeff[1]*np.log10(df.iloc[:,0]) + coeff[2])
    return df,coeff,thrs
def prob_density_L(df, MFI_range):
    equation = df.apply(lambda col: col.dropna()[col.dropna().astype(str).str.contains('equation|molecule|molec', case=False)] if col.dtype == 'object' else pd.Series(), axis=0).stack().values
    coeff = coefficients(equation[0])
    df = df.loc[(df.iloc[:,0] > MFI_range[0]) & (df.iloc[:,0] <= MFI_range[1])& (df.iloc[:,1] > 0)].reset_index(drop=True)
    df = df.iloc[:,:2]
    df['nmbr_mcule'] = coeff[0]**(coeff[1]*np.log10(df.iloc[:,0]) + coeff[2])
    return df,coeff
def adding_prob_dens(df, MFI_range=None,Cell_type=None,coeff=None,thrs = None):
    bin=5
    if Cell_type == "NK_cell":
        df,coeff,thrs = prob_density_R(df, MFI_range)
    elif Cell_type == "Tumor_cell":
        df,coeff = prob_density_L(df, MFI_range)
    else:
        raise ValueError("Cell _type must be either 'NK_cell' or 'Tumor_cell'")
    total = df.iloc[:,1].sum() if df.iloc[:,1].sum() > 0 else 1
    if thrs !=None:
        pive_thrs = coeff[0]**(coeff[1]*np.log10(thrs) + coeff[2])
        below_thrs = np.histogram(df['nmbr_mcule'][df['nmbr_mcule']<pive_thrs],bins=1,weights=df.iloc[:,1][df['nmbr_mcule']<pive_thrs])
        abv_thrs = np.histogram(df['nmbr_mcule'][df['nmbr_mcule']>=pive_thrs],bins=4,weights=df.iloc[:,1][df['nmbr_mcule']>=pive_thrs])
        below_thrs[1][1] = (below_thrs[1][1]+abv_thrs[1][0])/2
        prob = np.concatenate((below_thrs[0],abv_thrs[0]))
        bins = np.concatenate((below_thrs[1],abv_thrs[1][1:]))
        prob = prob/total
    else:
        prob,bins = np.histogram(df.iloc[:,2],bins=bin,weights=df.iloc[:,1])
        prob = prob/total
    nmbr = (bins[:-1] + bins[1:])/2
    if thrs:
        nmbr[0] = 0.0
    #return [df, bins, prob] # Just to see R-L distribution
    nmbr = np.array([nmbr@prob])
    prob=np.array([1.0])
    return [nmbr, prob, bins, df,thrs]
