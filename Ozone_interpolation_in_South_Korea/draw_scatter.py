
import numpy as np

# sklearn
from sklearn.metrics import mean_squared_error, r2_score

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns

# own package
import settings
from postprocessing_evaluation_tools import get_characteristics_df
from postprocessing_evaluation_tools import index_of_agreement

"""
A heatmap, only isolated gaps, only correlated gaps, summary.
"""
print('true versus imputed...')
dicts = [{'identifier': '_single','corr_list': [False]}]
for dict_ in dicts:
    print('\n', dict_['identifier'], '\n')

    # prepare scatter true vs. imputed
    corr_list = dict_['corr_list']
    df = get_characteristics_df()
    y_tru_short = df[(df.test_mask) &
                     (df.gap_len<=2) &
                     (df.correlated.isin(corr_list))].y_true
    y_imp_short = df[(df.test_mask) &
                     (df.gap_len<=2) &
                     (df.correlated.isin(corr_list))].y_imputed
    y_tru_long = df[(df.test_mask) &
                     (df.gap_len>2) &
                     (df.correlated.isin(corr_list))].y_true
    y_imp_long = df[(df.test_mask) &
                     (df.gap_len>2) &
                     (df.correlated.isin(corr_list))].y_imputed
    y_tru = df[df.test_mask].y_true
    y_imp = df[df.test_mask].y_imputed

    # print statistics, only for summary
    for tru, imp, what in [(y_tru_short, y_imp_short, 'short'),
                           (y_tru_long, y_imp_long, 'long'),
                           (y_tru, y_imp, 'summary') ]:
        if (dict_['identifier'] != '') & (what== 'summary'):
            continue
        r2 = r2_score(tru, imp)
        rmse = (mean_squared_error(tru, imp))**.5
        d = index_of_agreement(tru, imp)
        print(what)
        print(f'r2: {r2:.2f}')
        print(f'rmse: {rmse:.2f}')
        print(f'd: {d:.2f}\n')

    # plot style
    plt.style.use('seaborn-darkgrid')
    sns.set(rc={'axes.facecolor':'whitesmoke'})

    # scatter true vs. imputed
    fig, ax = plt.subplots(1, 2)
    min_val = -7.
    max_val = 115

    # Construct 2D histogram from data using the 'plasma' colormap
    norm = LogNorm(vmin=1, vmax=600) # Adjust scale of the heatmap
    colormap = 'autumn'  # autumn, winter, Blues_r, PuBu_r

    ax[0].hist2d(y_tru_short,
                 y_imp_short,
                 bins=(np.arange(min_val, max_val, 1.0),
                       np.arange(min_val, max_val, 1.0)),
                 norm=norm,
                 # cmap="Blues_r"
                 cmap=colormap
                )

    ax[1].hist2d(y_tru_long,
                 y_imp_long,
                 bins=(np.arange(min_val, max_val, 1.0),
                       np.arange(min_val, max_val, 1.0)),
                 norm=norm,
                 # cmap="Blues_r"
                 cmap=colormap
                )

    for ax_ in [0, 1]:
        ax[ax_].plot([min_val, max_val],
                     [min_val, max_val],
                     linestyle='--',
                     lw=1.2,
                     color='gray',
                     alpha=.55,
                     zorder=1,
                     )
        ax[ax_].set_aspect('equal', adjustable='box')
        ax[ax_].set_xlim(min_val, max_val)
        ax[ax_].set_ylim(min_val, max_val)
        ax[ax_].set_xlabel('true O3 [ppb]')
        #ax[ax_].set_ylabel('imputed O3 [ppb]')
        ax[ax_].grid(True)
    fig.set_size_inches(10, 5)
    ax[0].set_ylabel('imputed O3 [ppb]')

    # Plot a colorbar with label.
    m = cm.ScalarMappable(cmap=colormap,norm=norm)
    m.set_array([])
    cb = plt.colorbar(m, ax=ax,shrink=0.4)
    cb.set_label('Number of entries')

    # save
    id_ = dict_['identifier']
    true_vs_imp_heat_path = settings.output_dir + \
                            f'true_vs_imp_heatmap{id_}.png'
    plt.savefig(true_vs_imp_heat_path, dpi=500)
    print(f'written to {true_vs_imp_heat_path}')
    plt.close()