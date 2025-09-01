import numpy as np
import pandas as pd
import random
import re
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path
data_path =Path.cwd().parents[0]/'data/In_Vivo_Mv411'
def plot(i, days, mean_data, std_data, axes, label, color):
    axes = axes.flatten()
    mult = [1.1, 1.5]
    mean_data = np.where(mean_data <= 0, 0.01, mean_data)

    ax = axes[i]  # Get the subplot for this cage
    ax.errorbar(days, mean_data,#std_data,
                fmt='^-', 
                markersize=10, 
                markerfacecolor='lightgray', 
                markeredgewidth=3.5, 
                markeredgecolor=color,
                linewidth=2.5, 
                color=color, 
                elinewidth=3.5, 
                capsize=8, 
                capthick=15,
                ecolor=color, 
                alpha=0.99, 
                label=label)
    
    ax.set_yscale('log')

    max_mean_value = mean_data.max()#skipna=True)
    min_mean_value = mean_data.min()#skipna=True)
    # if not np.isnan(max_mean_value) and not np.isnan(min_mean_value):
    #     ax.set_ylim(max(0.01, min_mean_value * 0.1), max_mean_value * mult[i])
    ax.tick_params(axis='both', which='major', labelsize=12,  # Font size of tick labels
                   length=6, width=1.5)  # Length and width of tick marks
    ax.tick_params(axis='both', which='minor', labelsize=10, 
                   length=4, width=1)  # Minor ticks (optional, useful for log scale)

    # Customize text sizes
    #ax.set_title(Cage_names[0], fontsize=16)  # Title font size


    ax.set_yticks([10**5,10**6,10**7])
    ax.set_ylabel("Data", fontsize=14)  # Y-axis label font size
    ax.set_xlabel("Days", fontsize=14)  # X-axis label font size (added for completeness)
    ax.legend(fontsize=12)
def mean_std_three_cages():
    """
    Reads mouse data from an Excel file, processes it, and plots the mean and standard deviation of the data for three cages.
    (i) for WT it assumes cages 1, 2, and 3, and make the avearage of the three cages,
    (ii) for CAR it assumes cages 4, 5, and 6,
    (iii) for tumor only it assumes cages 7, 8, and 9.
    Returns:
        tuple: A tuple containing the days-4 (shifted), cage means, and cage standard deviations.
    """
    random.seed(42)  # Set seed for reproducibility
    df = pd.read_excel(f'{data_path}/Mouse_Data_CD33.xlsx')
    days = df.iloc[:, 1].str.extract('(\d+)').astype(int)
    num_cages = 3
    num_cols = (df.shape[1] - 1) // num_cages
    cage_data = []
    cage_mean = []
    cage_std = []
    for i in range(num_cages):
        df_new = df.iloc[:, i*num_cols+2:(i+1)*num_cols+2]
        cage_data.append(df_new.apply(pd.to_numeric, errors='coerce'))
        cage_mean.append(cage_data[i].mean(axis=1).values)
        cage_std.append(cage_data[i].std(axis=1).values)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    plt.rcParams['axes.linewidth'] = 2
    axes = axes.flatten()
    init_time = 0
    time = 6
    cage_mean = np.array(cage_mean)[:,init_time:time]
    cage_std = np.array(cage_std)[:,init_time:time]
    #print(days)
    days = days.values.flatten()[init_time:time]
    #print(days)
    plot(0,days[:],cage_mean[0], cage_std[0],axes,label='WT: Cage 1,2,3',color='gray')
    plot(0,days,cage_mean[1], cage_std[1],axes,label='CD33 CAR: Cage 4,5,6',color='blue')
    plot(1,days,cage_mean[2], cage_std[2],axes,label='Tumor only: Cage 7,8,9',color='black')
    plt.tight_layout()
    #plt.savefig('Mouse_data_mean.png')
    plt.savefig('Mouse_data_mean_log_scale.png')
    plt.show()
    return days-days[0],cage_mean,cage_std
########
def R_L_data(NK_cell, Tumor_cell,limts):
    data_R_type = {}
    data_L_type = {}
    keys = ['CAR','LFA1','KIR']
    for i,key in enumerate(keys):
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
    equation = df.apply(lambda col: col.dropna()[col.dropna().astype(str).str.contains('molecules', case=False)] if col.dtype == 'object' else pd.Series(), axis=0).stack().values
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
    return [nmbr, prob, bins, df,thrs]
