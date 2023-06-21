"""
The `plotting` script contains code to plot the results `janet` collects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lightkurve as lk
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.stats

sns.set_palette('colorblind')
sns.set_context('poster')

from .utils import _gaussian_fn, _safety
_label_fontsize=24
cmap = sns.color_palette('viridis', 8)
colmap = sns.color_palette('colorblind', 8)
binfactor = 20


def plot_tpf(j, fig, ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    sector0 = j.sectors[0]
    sectorlist0 = j.sectorlist[0]
    tpfs = j.void[f'tpfs_{sector0}']

    ax.set_title(f'Frame 0 Sector {sectorlist0}')
    ax.imshow(np.log10(tpfs[0][0].flux.value.T), zorder=1, origin='lower')

def plot_lcs(j, fig, ax):
    text = ''
    if len(j.sectors) > 1:
        for s in j.sectors[:-1]:
            text += f'{s}, '
        text = text[:-2]
        text += f' & {j.sectors[-1]}'
    else:
        text = j.sectors[0]
    ax.set_title(f'Corrected TESS-SIP LC, Sectors: {text}. Normalised, outliers removed.')

    xstep = 0
    xlabels = []
    xlocs = []
    for s in j.sectors:
        lc = j.void[f'{j.pl}lc_{s}']
        xvals = lc.time.value - lc.time.value.min() + xstep
        ax.scatter(xvals, lc.flux, label=f'Sector(s) {s}', s=1)
        xstep = xvals.max()
        if s != j.sectors[-1]:
            ax.axvline(xstep, c='k', ls='-', lw=3, zorder=200)
        xlabels.append(np.nanpercentile(lc.time.value, [25, 50, 75]))
        xlocs.append(np.round(np.nanpercentile(xvals, [15, 50, 85]),2))

        # binned = lk.LightCurve(time=xvals, flux=lc.flux).bin(bins = int(len(lc)/binfactor))
        if s == j.sectors[-1]:
            label = 'Smoothed LC'
        else:
            label = None

        sd = np.sqrt(len(lc))
        fsmoo = gaussian_filter1d(lc.flux.value, sigma = sd/4, mode='reflect')
        ax.plot(xvals, fsmoo, zorder=104, lw=5, color =cmap[4], label=label)
        ax.plot(xvals, fsmoo, zorder=103, lw=10, c='w')

    ax.set_xticks(np.array(xlocs).flatten())
    ax.set_xticklabels(np.array(xlabels).flatten().astype(int))
    ax.legend(loc='best')
    ax.set_xlabel('Normalised Time [JD]')
    ax.set_xlim(0, xstep)
    ax.set_ylabel('Normalised Flux')

def plot_periodograms(j, fig, ax):
    best_SIP = j.results.loc['best', 's_SIP']

    for s in j.sectors:
        j.void[f'SIP_pg_{s}'].plot(ax=ax, view='period',
        label=f'Sector(s) {s}', lw=2, zorder=2)
    ax.axvline(j.results.loc["best", "SIP"], color=cmap[4], lw=5, ls='--', zorder=1, label=f'P = {j.results.loc["best", "SIP"]:.2f} d')
    ax.set_xlim(j.void[f'SIP_pg_{best_SIP}'].period.min().value, j.void[f'SIP_pg_{best_SIP}'].period.max().value)
    ax.set_ylim(0)
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_xscale('log')
    ax.set_title('All SIP Periodograms')

def plot_periodogram_fit(j, fig, ax):
    best_SIP = j.results.loc['best', 's_SIP']

    text = f'Sector(s) {best_SIP}'

    ax.get_yaxis().set_visible(False)
    ax.plot(j.void[f'p_{best_SIP}'],
            _gaussian_fn(j.void[f'p_{best_SIP}'], *j.void[f'popt_{best_SIP}']), ls='--', lw=10, color=cmap[5], zorder=2,
            label = rf'$\sigma$ = {j.results.loc["best", "e_SIP"]:.2f} d')
    ax.set_xlim(j.void[f'popt_{best_SIP}'][0] - 5*j.void[f'popt_{best_SIP}'][1],
                    j.void[f'popt_{best_SIP}'][0] + 5*j.void[f'popt_{best_SIP}'][1])

    for s in j.sectors:
        j.void[f'SIP_pg_{s}'].plot(ax=ax,lw=2, zorder=0)
    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_xlabel('Period [d]')
    ax.set_title(f'Fit to SIP {text}')

def plot_fold(j, fig, ax):
    xstep = 0
    xlabels = []
    xlocs = []
    for s in j.sectors:
        period = j.results.loc['best', 'SIP']

        lc = j.void[f'{j.pl}lc_{s}'].fold(period=period)
        xvals = lc.time.value - lc.time.value.min() + xstep
        ax.scatter(xvals, lc.flux, s=2, label=f'Sector(s) {s} Folded')
        xstep = xvals.max()
        if s != j.sectors[-1]:
            ax.axvline(xstep, c='k', ls='-', lw=3, zorder=10000)
        xlabels.append(np.round(np.nanpercentile(lc.time.value, [25, 50, 75]),2))
        xlocs.append(np.round(np.nanpercentile(xvals, [15, 50, 85]), 2))

        # Plot the smoothed version
        # Plot approval of the sector folded on the best result using colour
        if s == j.sectors[-1]:
            label = 'Smoothed LC'
        else:
            label = None

        sd = np.sqrt(len(lc))
        fsmoo = gaussian_filter1d(lc.flux.value, sigma = sd, mode='reflect')
        p2p = np.diff([np.nanmin(fsmoo), np.nanmax(fsmoo)])
        mad = scipy.stats.median_abs_deviation(lc.flux.value /
                                        gaussian_filter1d(lc.flux.value,
                                            sigma = sd, mode = 'nearest'))

        check = p2p > 2*mad
        if check:
            linecol = cmap[4]
            ls = '-'
        else:
            linecol = 'r'
            ls = '--'
        ax.plot(xvals, fsmoo, lw=10, c='k', zorder=103)
        ax.plot(xvals, fsmoo, lw=5, c=linecol, ls=ls, label=label, zorder=104)

    ax.set_xlim(0, xstep)
    ax.set_xticks(np.array(xlocs).flatten())
    ax.set_xticklabels(np.array(xlabels).flatten())

    ax.legend(loc='best')
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_title(rf'All Sectors folded on Best Period: {period:.2f} $\pm$ {j.results.loc["best", "e_overall"]:.2f} d')
    ax.axhline(1.00, lw=5, ls='--', c='k', zorder=100)

def plot_SIP(j):
   
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2,3, figure=fig)

    ax00 = fig.add_subplot(gs[0,0])
    plot_tpf(j, fig, ax00)

    # Plot all LCs
    ax01 = fig.add_subplot(gs[0, 1:])
    plot_lcs(j, fig, ax01)

    # Plot all periodograms
    ax10 = fig.add_subplot(gs[1, :2])
    plot_periodograms(j, fig, ax10)

    # Plot Sector PG Fit
    if np.isfinite(j.results.loc['best','SIP']):
        ax11 = fig.add_subplot(gs[1, 2:], sharey=ax10)
        plot_periodogram_fit(j, fig, ax11)
        ax11.minorticks_on()

    # Plot the phase folded light curve
    ax2 = fig.add_subplot(gs[2, :])
    plot_fold(j, fig, ax2)

    # Polish
    if j.sectors[0] != 0:
        ax00.minorticks_on()
    ax10.minorticks_on()
    ax2.minorticks_on()
    fig.tight_layout()

    fig.suptitle(f'Gaia ID: {j.gaiaid} - Pipeline: {j.pipeline}', fontsize=30)
    plt.subplots_adjust(top=0.95)

    plt.savefig(f'{j.output_path}/{j.gaiaid}/{j.pl}_output.pdf')
    plt.savefig(f'{j.output_path}/{j.gaiaid}/{j.pl}_output.png', dpi = 300)
