import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
def plot(model_data,mean_data,std_data,ET_ratio,yticks,colors,out_file):
    ET_ratio_num = np.arange(len(ET_ratio))
    plt.figure(figsize=(13,9))
    plt.rcParams['axes.linewidth'] = 8
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=19, length=8, width=5)
    ls = [(1,(1,1)),(5, (10, 3)), (0,(2,2)), (0, (3, 1, 1, 1))]
    m_size = 22
    lw = 5.5
    for i in range(len(mean_data)):
        plt.plot(ET_ratio_num, model_data[i], marker='o', markersize=m_size, markerfacecolor='white', markeredgewidth=lw, markeredgecolor=colors[i],color = colors[i],alpha=0.95,lw=lw,ls = ls[0],label="Model-fit-Gen")
        #plt.plot(ET_ratio_num, model_data[1], marker='o', markersize=m_size, markerfacecolor='white', markeredgewidth=lw, markeredgecolor = colors[1], color = colors[1],alpha=0.95,lw=lw,ls = ls[0],label="Model-fit-WT")


        plt.errorbar(ET_ratio_num, mean_data[i], yerr = std_data[i], 
                    fmt='^', markersize=m_size, markerfacecolor=colors[2+i], markeredgewidth=lw, markeredgecolor=colors[i],
                    elinewidth=lw, capsize=10, capthick=20,
                    ecolor=colors[i],alpha=0.95,label="data-CAR")
    # plt.errorbar(ET_ratio_num, mean_data[1], yerr = std_data[1], 
    #             fmt='^', markersize=m_size, markerfacecolor=colors[3], markeredgewidth=lw, markeredgecolor=colors[1],
    #             elinewidth=lw, capsize=10, capthick=20,
    #             ecolor=colors[0],alpha=0.95,label="data-WT")
    t_size = 45
    plt.xticks(ET_ratio_num,ET_ratio,fontname="Arial",fontsize = t_size, rotation=25)
    plt.yticks(yticks,fontname="Arial",fontsize = t_size)
    plt.tight_layout(pad=1.0)
    #plt.xlabel('ET_ratio',fontsize=25)
    #plt.ylabel('%Av-Specific Lysis',fontsize=30)
    #plt.title('Kasumi1',fontsize=30)
    plt.legend(bbox_to_anchor=(0.98, 0.98),fontsize=22, loc='upper right', labelcolor='white')
    #plt.savefig(out_file, bbox_inches = 'tight')
    plt.show()

def R2_plot(y_data, y_pred,title,out_file):
    fig = plt.figure(figsize=(8,4))
    plt.rcParams['axes.linewidth'] = 2.5
    rows,cols = 1, len(y_data)
    for i in range(1,rows*cols+1):
        fig.add_subplot(rows,cols,i)
        plt.plot(y_data[i-1],y_pred[i-1],'o')
        plt.xticks([10,50,90],fontsize=15)
        plt.yticks([10,50,90],fontsize=15)
        #plt.title(tit[i-1],fontsize=12)
        plt.xlabel('data',fontsize=12)
        plt.ylabel('Prediction',fontsize=12)
        plt.title(f'$R^2 = {r2_score(y_data[i-1],y_pred[i-1]):.2f}$ ({title[i-1]})',fontsize =13)
        plt.plot(np.arange(0,100),np.arange(0,100),color='gray',lw=2.5)
    plt.tight_layout()
    #plt.savefig(out_file)
    plt.show()