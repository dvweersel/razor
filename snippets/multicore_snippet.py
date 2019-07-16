https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations

import pandas as pd
import numpy as np
import multiprocessing
from functools import partial

def _df_split(tup_arg, **kwargs):
    split_ind, df_split, df_f_name = tup_arg
    return (split_ind, getattr(df_split, df_f_name)(**kwargs))

def df_multi_core(df, df_f_name, subset=None, njobs=-1, **kwargs):
    if njobs == -1:
        njobs = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=njobs)

    try:
        splits = np.array_split(df[subset], njobs)
    except ValueError:
        splits = np.array_split(df, njobs)

    pool_data = [(split_ind, df_split, df_f_name) for split_ind, df_split in enumerate(splits)]
    results = pool.map(partial(_df_split, **kwargs), pool_data)
    pool.close()
    pool.join()
    results = sorted(results, key=lambda x:x[0])
    results = pd.concat([split[1] for split in results])
    return results

# Plot a pareto graph for a continious variable with a matching y-axis
def pareto(df, column):
    _df = df.sort_values(column, ascending=False).reset_index(drop=True).reset_index()
    _df['percentage'] = (_df['index'] / _df['index'].count()) * 100

    column_norm = column + '_Norm'
    column_cum = column + '_Cum'

    _df[column_norm] = _df[column] / df[column].sum()
    _df[column_cum] = _df[column].cumsum() *100 / df[column].sum()

    fig, ax1 = plt.subplots()

    _df.plot(x='percentage', y=column, ax=ax1, color='red', legend=False, ylim=(0, _df[column].max()))
    ax2 = ax1.twinx()

    _df.plot(x='percentage', y=column_cum, ax=ax2, color='blue', legend=False, grid=True, ylim=(0,100))

    ax2.set_yticks(np.linspace(0, 100, 11))
    
    ax1_ticks = np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], 11)
    ax1.set_yticks(ax1_ticks)
    
    ax1.set_xticks(np.linspace(0, 100, 11))

    plt.show()