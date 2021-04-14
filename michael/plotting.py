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

    fig = plt.figure(figsize=(20, 30))
    gs = GridSpec(4,3, figure=fig)

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
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'clc_{s}'].plot(ax=ax01, lw=1)
    else:
        j.void[f'clc_all'].plot(ax=ax01, lw=1, c='k')

    ax01.set_xlim(j.void[f'clc_all'].time.min().value, j.void[f'clc_all'].time.max().value)

    # Plot all periodograms
    ax10 = fig.add_subplot(gs[1, :2])
    j.void[f'pg_all'].plot(ax=ax10, view='period', label=f'All Sectors',lw=1, zorder=2, c='k')
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'pg_{s}'].plot(ax=ax10, view='period',
            label=f'Sector {s}', lw=1, zorder=2)
    ax10.axvline(j.void['popt_all'][0], c=cmap[4], lw=5, ls='--', zorder=1, label=f'P = {j.results.loc["all", "SLS"]:.2f} d')
    ax10.set_xlim(j.void[f'pg_all'].period.min().value, j.void[f'pg_all'].period.max().value)
    ax10.set_ylim(0)
    ax10.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax10.set_xscale('log')
    ax10.set_title('All Lomb Scargle Periodograms')

    # Plot Sector All PG Fit
    ax11 = fig.add_subplot(gs[1, 2:], sharey=ax10)
    ax11.get_yaxis().set_visible(False)
    ax11.plot(j.void['p_all'], j.void['P_all'], lw=1, c='k', zorder=1)
    ax11.plot(j.void['p_all'],
            _gaussian_fn(j.void['p_all'], *j.void['popt_all']), ls='--', lw=10, c=cmap[5], zorder=2,
            label = rf'$\sigma$ = {j.results.loc["all", "e_SLS"]:.2f} d')
    ax11.set_xlim(j.void['popt_all'][0] - 5*j.void['popt_all'][1],
                    j.void['popt_all'][0] + 5*j.void['popt_all'][1])
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'pg_{s}'].plot(ax=ax11,lw=1, zorder=0)
    ax11.legend(loc='best', fontsize=_label_fontsize)
    ax11.set_xlabel('Period [d]')
    ax11.set_title('Fit to LSP All Sectors')

    # Wavelet contourfplot
    axw1 = fig.add_subplot(gs[2, :2])
    c = axw1.contourf(j.void['wt'].taus, 1./j.void['wt'].nus, j.void['wwz'])
    # axw1.set_yscale('log')
    axw1.set_ylabel('Period [d]')
    axw1.set_xlabel('Time [JD]')
    fig.colorbar(c, ax=axw1, label='WWZ', pad=.01, aspect=60)
    if len(j.sectors) >= 2:
        for s in j.sectors[1:]:
            axw1.axvline(j.void[f'clc_{s}'].time.min().value, c='w', ls='-.', lw=3)
        for s in j.sectors:
            axw1.text(j.void[f'clc_{s}'].time.min().value+1, (1./j.void['wt'].nus).max() * 0.925, f'S{s}', c='w', weight='bold')
    axw1.axhline(j.results.loc['all', 'SW'], ls='--', lw = 3, c='w', label=f'P = {j.results.loc["all", "SW"]:.2f} d')
    axw1.set_title('Wavelet Transform')
    axw1.legend(loc='best', fontsize=_label_fontsize)

    # Collapsed Wavelet and fit
    axw2 = fig.add_subplot(gs[2, 2:])
    taus = j.void['wt'].taus
    w =  np.sum(j.void['wwz'], axis=1)
    w /= w.max()
    p = 1/j.void['wt'].nus
    axw2.plot(p, w, lw=1, c='k')
    if len(j.sectors) >= 2:
        for s in j.sectors:
            time = j.void[f'clc_{s}'].time.value
            s = (taus >= time.min()) & (taus <= time.max())
            ws = np.sum(j.void['wwz'][:, s], axis=1)
            ws /= ws.max()
            axw2.plot(p, ws, lw=1)

    axw2.plot(p, _gaussian_fn(p, *j.void['wavelet_popt']), ls='--', lw=10, c=cmap[5],
                label = rf'$\sigma$ = {j.results.loc["all", "e_SW"]:.2f} d')
    axw2.set_ylabel('Normalized Summed WWZ')
    axw2.set_xlabel('Period [d]')
    axw2.set_title('Fit to Summed WWZ')
    axw2.legend(loc='best', fontsize=_label_fontsize)
    axw2.set_ylim(0, 1.05)
    axw2.set_xlim(j.void['wavelet_popt'][0] - 5*j.void['wavelet_popt'][1],
                    j.void['wavelet_popt'][0] + 5*j.void['wavelet_popt'][1])

    # Plot the results compared

    # Plot the phase folded light curve
    ax2 = fig.add_subplot(gs[3, :2])
    fold = j.void['clc_all'].fold(period=j.results.loc['all', 'SLS'])
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'clc_{s}'].fold(period=j.results.loc['all', 'SLS']).scatter(s=75, label=f'Sector {s} Folded', ax=ax2, zorder=1)
    else:
        fold.scatter(ax=ax2, c='k', s=75, label='Folded LC', zorder=1)
    fold.bin(bins=int(len(fold)/50)).plot(ax=ax2, zorder=4, lw=5, c=cmap[4], label='Binned LC')
    fold.bin(bins=int(len(fold)/50)).plot(ax=ax2, zorder=3, lw=10, c='w')

    ax2.axhline(1., c='k', zorder=2, ls='-')
    ax2.set_xlim(fold.phase.min().value, fold.phase.max().value)
    ax2.legend(loc='best', fontsize=_label_fontsize)
    ax2.set_title(rf'All Sectors folded on Period: {j.results.loc["all", "SLS"]:.2f} $\pm$ {j.results.loc["all", "e_SLS"]:.2f} days')

    # Polish
    ax00.minorticks_on()
    ax01.minorticks_on()
    ax10.minorticks_on()
    ax11.minorticks_on()
    axw1.minorticks_on()
    axw2.minorticks_on()
    ax2.minorticks_on()
    fig.tight_layout()


    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.pdf', rasterized=True)
    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.png', dpi = 300)
