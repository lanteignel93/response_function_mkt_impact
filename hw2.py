import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
ROOT_DIR = os.getcwd()
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
DATA_PATH = os.path.join(ROOT_DIR, "data")
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

plt.style.use('seaborn')

data_v1 = pd.read_csv(DATA_PATH + '\\' + 'pp1_md_201607_201607.csv')
data_v2 = pd.read_csv(DATA_PATH + '\\' + 'pp1_md_201608_201608.csv')
df = pd.concat([data_v1,data_v2]).dropna()
df.reset_index(inplace=True)
# This is to fix the rows where the bid is 0 or when the ask is 0
df = df[(df['BP1'] > 0) & (df['SP1'] > 0)]

def compute_rls(df, lags):
    return np.array([(((df.VWAP.shift(-x) - df.midQ) * df.Sign)/(df.SP1-df.BP1)).dropna().mean() for x in lags])

# def compute_rls(df, lags):
#     return np.array([((df.VWAP.shift(-x) - df.midQ) * df.Sign).dropna().mean() for x in lags])

def plot_q2():
    fig, ax = plt.subplots(1, figsize = (12,8))
    x = np.arange(0,501,1)
    y = compute_rls(df, x)
    ax.plot(x,y)
    ax.set_title("Response Function $R_l$ for $0 \leq l \leq 500$", fontsize = 18)
    ax.set_ylabel("$R_l$", fontsize = 14)
    ax.set_xlabel("$Lags$", fontsize = 14)
    save_fig('hw2_q2')
    plt.show()

plot_q2()

def plot_q3():
    fig, ax = plt.subplots(5,2, figsize = (12,16))
    v = [0,2,5,10,15,20,30,40,55,90,1000000]
    x = np.arange(0,501,1)
    group_dic = {}
    ax_dic = {}
    for k in range(1,11,1):
        group_dic[f'Group {k}'] = (v[k-1],v[k])
        if k < 6:
            ax_dic[k] = ax[k-1,0]
        else:
            ax_dic[k] = ax[k-6,1]
    plt.suptitle("$R_l$ for different groups of trade sizes", fontsize = 16)
    for i in range(1,11,1):
        trade_size = group_dic['Group {}'.format(i)]
        tmp = df[(df.Size >= trade_size[0]) & (df.Size <= trade_size[1])]
        y = compute_rls(tmp, x)
        ax_dic[i].plot(x,y)
        ax_dic[i].set_title(f"Response Function $R_l$ for Group {i} with $v_1=${trade_size[0]} and $v_2=${trade_size[1]}", fontsize = 12)
        ax_dic[i].set_ylabel("$R_l$", fontsize = 10)
        ax_dic[i].set_xlabel("$Lags$", fontsize = 10)

    plt.tight_layout()
    save_fig('hw2_q3')
    plt.show()

plot_q3()

def plot_q4():
    fig, ax = plt.subplots(6,2, figsize = (12,18))
    v = [0,2,5,10,15,20,30,40,55,90,1000000]
    log_average_trade = np.array([np.log(df[(df.Size >= v[x-1]) & (df.Size <= v[x])].Size.mean()) for x in range(1,11,1)]).reshape(len(v)-1,1)
    lags = [10,20,30,40,50,75,100,125,150,175,200,250]
    group_dic = {}
    ax_dic = {}

    for k in range(12):
        y = []
        for i in range(10):
            tmp = df[(df.Size >= v[i]) & (df.Size <= v[i+1])]
            y_tmp = np.log(compute_rls(tmp, [lags[k]]))
            y.append(y_tmp)
        group_dic[f'Lag: {lags[k]}'] = y
        if k < 6:
            ax_dic[k] = ax[k-1,0]
        else:
            ax_dic[k] = ax[k-6,1]

    plt.suptitle(r"Log Responses $\log \left(\tilde{R}_l|_{v_i < V_i < v_{i+1}} \right)$ vs Log Average Trade Size $\log(V_i)$", fontsize = 16)
    for i in range(12):
        y = np.array(group_dic[f'Lag: {lags[i]}'])
        x = np.hstack((np.ones(10).reshape(10,1),log_average_trade))
        beta = (np.linalg.inv(x.T@x) @ x.T @ y)
        x_fitted = np.linspace(log_average_trade[0], log_average_trade[-1], 1000)
        y_fitted = beta[0][0] + beta[1][0] * x_fitted
        ax_dic[i].scatter(log_average_trade,y)
        ax_dic[i].plot(x_fitted, y_fitted, ls='-.', lw = 1, color = 'red',label = 'Fitted Slope: {:.3f}'.format(beta[1][0]))
        ax_dic[i].set_title(f"$l=$ {lags[i]}", fontsize = 12)
        ax_dic[i].set_ylabel("Log Response $\log(R_l)$", fontsize = 10)
        ax_dic[i].set_xlabel("Log Average Trade Size", fontsize = 10)
        ax_dic[i].legend(fontsize = 14)
    plt.tight_layout()
    save_fig('hw2_q4')
    plt.show()

plot_q4()
