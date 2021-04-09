"""
The `plotting` script contains code to plot the results `janet` collects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lightkurve as lk
import seaborn as sns
sns.set_palette('colorblind')
sns.set_context('poster')

from .utils import _gaussian_fn, _safety
_label_fontsize=24

def plot(j):
    cmap = sns.color_palette('viridis', 8)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3,3, figure=fig)

    ## Plotting
    if len(j.sectors) == 1:
        sec0 = 'all'
        sectors = ['all']
    else:
        sec0 = j.sectors[0]
        sectors = [str(s) for s in j.sectors] + ['all']

    # Plot Sector 0 TPF
    ax00 = fig.add_subplot(gs[0, :1])
    ax00.set_title(f'Frame 0 Sector {j.sectors[0]}')
    ax00.set_xlabel('X')
    ax00.set_ylabel('Y')
    ax00.imshow(j.void[f'datum_{j.sectors[0]}'].tpf[0], zorder=1)
    pix = np.where(j.void[f'datum_{j.sectors[0]}'].aperture > 0)
    ax00.scatter(pix[0], pix[1], edgecolors='w', lw=5, marker=',', facecolors='none', s=600, zorder=2, label='Aperture')
    ax00.legend(loc='upper left', fontsize=_label_fontsize)

    # Plot Sector 0 LC
    ax01 = fig.add_subplot(gs[0, 1:])
    ax01.set_title(f'Full TESS LC, Sectors: {j.sectors}. Normalized, outliers removed.')
    j.void[f'clc_{sec0}'].plot(ax=ax01, c='k', lw=1)
    ax01.set_xlim(j.void[f'clc_{sec0}'].time.min().value, j.void[f'clc_{sec0}'].time.max().value)

    # Plot all periodograms
    ax10 = fig.add_subplot(gs[1, :2])
    j.void[f'pg_all'].plot(ax=ax10, view='period', label=f'All Sectors',lw=1, zorder=2, c='k')
    for s in j.sectors:
        j.void[f'pg_{s}'].plot(ax=ax10, view='period',
        label=f'Sector {s}', #lw=int(5/(j.results.loc[s, 'f_SLS']+1)),
        lw=1, zorder=2)
    ax10.axvline(j.void['popt_all'][0], c=cmap[4], lw=5, ls='--', zorder=1, label=f'P = {j.results.loc["all", "SLS"]:.2f} days')
    ax10.set_xlim(j.void[f'pg_{s}'].period.min().value, j.void[f'pg_{s}'].period.max().value)
    ax10.set_ylim(0)
    ax10.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax10.set_xscale('log')
    ax10.set_title('All Periodograms')

    # Plot Sector All PG Fit
    ax11 = fig.add_subplot(gs[1, 2:], sharey=ax10)
    ax11.get_yaxis().set_visible(False)
    ax11.plot(j.void['p_all'], j.void['P_all'], lw=1, c='k', zorder=1)
    ax11.plot(j.void['p_all'],
            _gaussian_fn(j.void['p_all'], *j.void['popt_all']), ls='--', lw=10, c=cmap[5], zorder=2)
    ax11.set_xlim(j.void['popt_all'][0] - 5*j.void['popt_all'][1],
                    j.void['popt_all'][0] + 5*j.void['popt_all'][1])
    for s in j.sectors:
        j.void[f'pg_{s}'].plot(ax=ax11,lw=1, zorder=0)
    ax11.set_xlabel('Period [d]')
    ax11.set_title('LS Fit to All Sectors')

    # Plot the phase folded light curve
    ax2 = fig.add_subplot(gs[2, :])
    fold = j.void['clc_all'].fold(period=j.results.loc['all', 'SLS'])
    fold.scatter(ax=ax2, c='k', s=5, label='Folded LC')
    fold.bin(bins=int(len(fold)/50)).plot(ax=ax2, zorder=2, lw=5, c=cmap[5], label='Binned LC')
    ax2.legend(loc='upper left', fontsize=_label_fontsize)
    ax2.set_title(f'All Sectors folded on Period: {j.results.loc["all", "SLS"]:.2f} days')

    # Polish
    ax00.minorticks_on()
    ax01.minorticks_on()
    ax10.minorticks_on()
    ax11.minorticks_on()
    ax2.minorticks_on()
    fig.tight_layout()


    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.pdf', rasterized=True)
    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.png', dpi = 300)
