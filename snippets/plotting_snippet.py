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