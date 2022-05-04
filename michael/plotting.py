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
cmap = sns.color_palette('viridis', 8)
colmap = sns.color_palette('colorblind', 8)


def plot_tpf(j, fig, ax):
    # Plot Sector 0 TPF
    if not j.override:
        sector0 = j.sectorlist[0]
        ax.set_title(f'Frame 0 Sector {sector0}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.imshow(np.log10(j.void[f'datum_{sector0}'].tpf[0]), zorder=1)
        pix = np.where(j.void[f'datum_{sector0}'].aperture > 0)
        ax.scatter(pix[0], pix[1], edgecolors='w', lw=5, marker=',', facecolors='none', s=600, zorder=2, label='Aperture')
        ax.legend(loc='upper left', fontsize=_label_fontsize)

def plot_lcs(j, fig, ax):

    # Sort out the text
    text = ''
    if len(j.sectors) > 1:
        for s in j.sectors[:-1]:
            text += f'{s}, '
        text = text[:-2]
        text += f' & {j.sectors[-1]}'
    else:
        text = j.sectors[0]
    ax.set_title(f'Full TESS LC, Sectors: {text}. Normalised, outliers removed.')

    xstep = 0
    xlabels = []
    xlocs = []
    for s in j.sectors:
        lc = j.void[f'clc_{s}']
        xvals = lc.time.value - lc.time.value.min() + xstep
        ax.scatter(xvals, lc.flux, label=f'Sector(s) {s}', s=1)
        xstep = xvals.max()
        if s != j.sectors[-1]:
            ax.axvline(xstep, c='k', ls='-', lw=3, zorder=10)
        xlabels.append(np.nanpercentile(lc.time.value, [25, 50, 75]))
        xlocs.append(np.round(np.nanpercentile(xvals, [15, 50, 85]),2))

    ax.set_xticks(np.array(xlocs).flatten())
    ax.set_xticklabels(np.array(xlabels).flatten().astype(int))
    ax.legend(loc='best')
    ax.set_xlabel('Normalised Time [JD]')
    ax.set_xlim(0, xstep)
    ax.set_ylabel('Normalised Flux')

def plot_periodograms(j, fig, ax):
    best_sls = j.results.loc['best', 's_SLS']

    for s in j.sectors:
        j.void[f'pg_{s}'].plot(ax=ax, view='period',
        label=f'Sector(s) {s}', lw=2, zorder=2)
    ax.axvline(j.results.loc["best", "SLS"], c=cmap[4], lw=5, ls='--', zorder=1, label=f'P = {j.results.loc["best", "SLS"]:.2f} d')
    ax.set_xlim(j.void[f'pg_{best_sls}'].period.min().value, j.void[f'pg_{best_sls}'].period.max().value)
    ax.set_ylim(0)
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_xscale('log')
    ax.set_title('All Lomb Scargle Periodograms')

def plot_periodogram_fit(j, fig, ax):
    best_sls = j.results.loc['best', 's_SLS']

    text = f'Sector(s) {best_sls}'

    ax.get_yaxis().set_visible(False)
    ax.plot(j.void[f'p_{best_sls}'],
            _gaussian_fn(j.void[f'p_{best_sls}'], *j.void[f'popt_{best_sls}']), ls='--', lw=10, c=cmap[5], zorder=2,
            label = rf'$\sigma$ = {j.results.loc["best", "e_SLS"]:.2f} d')
    ax.set_xlim(j.void[f'popt_{best_sls}'][0] - 5*j.void[f'popt_{best_sls}'][1],
                    j.void[f'popt_{best_sls}'][0] + 5*j.void[f'popt_{best_sls}'][1])

    for s in j.sectors:
        j.void[f'pg_{s}'].plot(ax=ax,lw=2, zorder=0)
    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_xlabel('Period [d]')
    ax.set_title(f'Fit to LSP {text}')

def plot_wavelet_contour(j, fig, ax):
    # if not j.gaps:
    #     ax.contourf(j.void['all_wt'].taus, 1./j.void['all_wt'].nus, j.void['all_wwz'])
    #
    #     if len(j.sectors) >= 2:
    #         for s in j.sectors[1:]:
    #             ax.axvline(j.void[f'clc_{s}'].time.min().value, c='w', ls='-.', lw=3)
    #         for s in j.sectors:
    #             ax.text(j.void[f'clc_{s}'].time.min().value+1, (1./j.void[f'all_wt'].nus).max() * 0.925, f'S{s}', c='w', weight='bold')
    #     ax.axhline(j.results.loc['best', 'SW'], ls='--', lw = 3, c='w', label=f'P = {j.results.loc["best", "SW"]:.2f} d')
    #
    # else:
    xstep = 0
    xlabels = []
    xlocs = []

    for s in j.sectors:
        xvals = j.void[f'{s}_wt'].taus - j.void[f'{s}_wt'].taus.min() + xstep
        ax.contourf(xvals, 1./j.void[f'{s}_wt'].nus, j.void[f'{s}_wwz'])
        ax.text(xstep+1, (1./j.void[f'{s}_wt'].nus).max() * 0.925, f'S{s}', c='w', weight='bold')
        xstep = xvals.max()
        if s != j.sectors[-1]:
            ax.axvline(xstep, c='w', ls='-', lw=10)
        xlabels.append(np.nanpercentile(j.void[f'{s}_wt'].taus, [25, 50, 75]))
        xlocs.append(np.round(np.nanpercentile(xvals, [15, 50, 85]),2))

    ax.axhline(j.results.loc['best', 'SW'], ls='--', lw = 3, c='w', label=f'P = {j.results.loc["best", "SW"]:.2f} d')
    ax.set_xticks(np.array(xlocs).flatten())
    ax.set_xticklabels(np.array(xlabels).flatten().astype(int))

    ax.set_title('Wavelet Transform')
    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_ylabel('Period [d]')
    ax.set_xlabel('Time [JD]')
    ax.set_ylim(j.period_range[0], j.period_range[1])

def plot_wavelet_fit(j, fig, ax):
    # if not j.gaps:
    #     taus = j.void['all_wt'].taus
    #     ws = np.sum(j.void['all_wwz'], axis=1)
    #     ws /= ws.max()
    #     p = 1/j.void['all_wt'].nus
    #     ax.plot(p, ws, lw=2, c='k')

    # if len(j.sectors) > 1:
    for s in j.sectors:
        taus = j.void[f'{s}_wt'].taus
        ws = np.sum(j.void[f'{s}_wwz'], axis=1)
        ws /= ws.max()
        p = 1/j.void[f'{s}_wt'].nus
        ax.plot(p, ws, lw=2)

    best_sw = j.results.loc['best', 's_SW']
    text = f'Sector(s) {best_sw}'

    taus = j.void[f'{best_sw}_wt'].taus
    w = np.sum(j.void[f'{best_sw}_wwz'], axis=1)
    w /= w.max()
    p = 1/j.void[f'{best_sw}_wt'].nus

    pp = np.linspace(p.min(), p.max(), 500)
    ax.plot(pp, _gaussian_fn(pp, *j.void[f'{best_sw}_wavelet_popt']), ls='--', lw=10, c=cmap[5],
                label = rf'$\sigma$ = {j.results.loc["best", "e_SW"]:.2f} d')
    ax.set_ylabel('Normalized Summed WWZ')
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Period [d]')
    ax.set_title(f'Fit to Summed WWZ {text}')
    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(j.void[f'{best_sw}_wavelet_popt'][0] - 5*j.void[f'{best_sw}_wavelet_popt'][1],
                    j.void[f'{best_sw}_wavelet_popt'][0] + 5*j.void[f'{best_sw}_wavelet_popt'][1])

def plot_cacf(j, fig, ax):
    # if not j.gaps:
    #     j.void['all_cacf'].plot(ax=ax, c='k', lw=1,  zorder=1, alpha=.5)
    #     ax.plot(j.void['all_cacf'].time.value, j.void['all_cacfsmoo'], c='k', lw=2, label='All Sectors', zorder=4)

    # if len(j.sectors) > 1:
    for idx, s in enumerate(j.sectors):
        j.void[f'{s}_cacf'].plot(ax=ax, c=colmap[idx], lw=1,  zorder=2, alpha=.5)
        ax.plot(j.void[f'{s}_cacf'].time.value, j.void[f'{s}_cacfsmoo'], c=colmap[idx], lw=2, label=f'Sector(s) {s}', zorder=5)


    ax.set_xlim(j.void['vizacf'].time.value.min(), j.void['vizacf'].time.value.max())

    ax.axvspan(j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'], color=cmap[6], zorder=2,
                label=f'P = {j.results.loc["best", "overall"]:.2f} $\pm$ {j.results.loc["best", "e_overall"]:.2f} d',
                alpha=.5)
    # ax.axvspan(j.results.loc['best', 'overall'] - 2*j.results.loc['best', 'e_overall'],
    #             j.results.loc['best', 'overall'] + 2*j.results.loc['best', 'e_overall'], color=cmap[7], zorder=1,
    #             label=r'$2\sigma$', alpha=.5)

    if np.isfinite(j.results.loc['best','CACF']):
        ax.axvline(j.results.loc['best', 'CACF'], c=cmap[3],
                    label = f'P = {j.results.loc["best", "CACF"]:.2f} d',
                    lw = 4, ls=':', zorder=10)

    ax.set_title("(Smoothed) Composite ACF for all Sectors")
    ax.set_ylabel('Normalised CACF')
    ax.axhline(0.01, label='Detection threshold', c='k', ls='--', zorder=0)
    ax.legend(loc='upper right',ncol = int(np.ceil(len(j.sectors)/4)))

def plot_cacf_fit(j, fig, ax):
    best_cacf = j.results.loc['best', 's_CACF']

    text = f'Sector(s) {best_cacf}'

    ax.get_yaxis().set_visible(False)
    ax.plot(j.void[f'{best_cacf}_cacf'].time.value,
            _gaussian_fn(j.void[f'{best_cacf}_cacf'].time.value,
                         *j.void[f'{best_cacf}_cacf_popt']), ls='--', lw=10, c=cmap[5], zorder=2,
            label = rf'$\sigma$ = {j.results.loc["best", "e_CACF"]:.2f} d')
    ax.set_xlim(j.void[f'{best_cacf}_cacf_popt'][0] - 5*j.void[f'{best_cacf}_cacf_popt'][1],
                    j.void[f'{best_cacf}_cacf_popt'][0] + 5*j.void[f'{best_cacf}_cacf_popt'][1])
    ax.set_ylim(0.)

    for idx, s in enumerate(j.sectors):
        ax.plot(j.void[f'{s}_cacf'].time.value, j.void[f'{s}_cacfsmoo'], c=colmap[idx], lw=2, zorder=0)

    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_xlabel('Period [d]')
    ax.set_title(f'Fit to CACF {text}')

def plot_acf(j, fig, ax):
    j.void['vizacf'].plot(ax=ax, c='k', zorder=3)
    ax.plot(j.void['vizacf'].time.value, j.void['acfsmoo'], lw=4, ls='--', c=cmap[3],
            label = 'Smoothed ACF', zorder=4)
    ax.set_ylim(j.void['vizacf'].flux.value.min(), j.void['vizacf'].flux.value.max()+0.1)
    ax.set_xlim(j.void['vizacf'].time.value.min(), j.void['vizacf'].time.value.max())

    if len(j.void['peaks']) >= 1:
        ax.axvline(j.void['vizacf'].time.value[j.void['peaks'][0]], c=cmap[3],
                        label = f'P = {j.results.loc["all", "ACF"]:.2f} d',
                        lw = 4, ls=':', zorder=5)
    ax.axvspan(j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'], color=cmap[6], zorder=2,
                label=f'P = {j.results.loc["best", "overall"]:.2f} $\pm$ {j.results.loc["best", "e_overall"]:.2f} d',
                alpha=.5)
    # ax.axvspan(j.results.loc['best', 'overall'] - 2*j.results.loc['best', 'e_overall'],
    #             j.results.loc['best', 'overall'] + 2*j.results.loc['best', 'e_overall'], color=cmap[7], zorder=1,
    #             label=r'$2\sigma$', alpha=.5)
    ax.set_title("Autocorrelation Function for all Sectors")
    ax.set_ylabel('Normalised ACF')
    ax.axhline(0.01, label='Detection threshold', c='k', zorder=0)
    ax.legend(loc='upper right')

def plot_comparison(j, fig, ax):
    ax.set_title('Period Estimates')
    ax.set_ylabel('Period [d]')
    ax.axhline(j.results.loc['all', 'ACF'],  label='ACF', c=cmap[3], ls=':', lw=5,
               zorder =1, alpha=.8)
    # Plot SLS
    if not j.gaps:
        xs = np.linspace(0.8, 1.2, len(j.sectors)+1)
        ax.errorbar(xs[-1], j.results.loc['all', 'SLS'],
                    yerr = j.results.loc['all', 'e_SLS'],
                    fmt='o', c='k')
    else:
        xs = np.linspace(0.8, 1.2, len(j.sectors))

    if len(j.sectors) > 1:
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'SLS'],
                        yerr = j.results.loc[sector, 'e_SLS'], fmt='o', c=colmap[idx])

    # Plot SW
    if not j.gaps:
        xs = np.linspace(1.8, 2.2, len(j.sectors)+1)
        ax.errorbar(xs[-1], j.results.loc['all', 'SW'], yerr = j.results.loc['all', 'e_SW'],
                    c='k', fmt='o')
    else:
        xs = np.linspace(1.8, 2.2, len(j.sectors))

    if len(j.sectors) > 1:
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'SW'],
                        yerr = j.results.loc[sector, 'e_SW'], fmt='o', c=colmap[idx])

    # Plot CACF
    if not j.gaps:
        xs = np.linspace(2.8, 3.2, len(j.sectors)+1)
        ax.errorbar(xs[-1], j.results.loc['all', 'CACF'], yerr = j.results.loc['all', 'e_CACF'],
                    c='k', fmt='o')
    else:
        xs = np.linspace(2.8, 3.2, len(j.sectors))

    if len(j.sectors) > 1:
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'CACF'],
                    yerr = j.results.loc[sector, 'e_CACF'], fmt='o', c=colmap[idx])

    labels = ['SLS', 'SW', 'CACF']
    x = [1., 2., 3.]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.axhspan(j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                color=cmap[6], alpha=.5, zorder=0,
               label = 'Best Period')
    if np.any(j.results[['SLS', 'SW', 'CACF','ACF']] > 1.9*j.results.loc['best', 'overall']) or\
        np.any(j.results[['SLS', 'SW', 'CACF','ACF']] < 0.6*j.results.loc['best', 'overall']):

        ax.axhspan(0.5*j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                    0.5*j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                    color=cmap[7], alpha=.5, zorder=0,
                   label = '2:1:2 Best')
        ax.axhspan(2*j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                    2*j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                    color=cmap[7], alpha=.5, zorder=0)
    ax.legend(loc='best')
    res = j.results.loc[j.results.index != 'best', ['SLS','SW', 'CACF']].to_numpy().flatten()
    err = j.results.loc[j.results.index != 'best', ['e_SLS', 'e_SW', 'e_CACF']].to_numpy().flatten()
    ax.set_ylim(0.9*np.nanmin(res-err), 1.1*np.nanmax(res+err))

def plot_fold(j, fig, ax):
    # if len(j.sectors) > 1:
    #     if not j.gaps:
    #         for idx, s in enumerate(j.sectors):
    #             j.void[f'clc_{s}'].fold(period=j.results.loc['best', 'overall']).scatter(
    #                     s=75, label=f'Sector {s} Folded', ax=ax, zorder=len(j.sectors) - idx)
    #         lc = j.void[f'clc_all'].fold(period=j.results.loc['best', 'overall'])
    #         binned = lc.bin(bins=int(len(lc)/50))
    #         binned.plot(ax=ax, zorder=104, lw=5, c=cmap[4], label='Binned LC')
    #         binned.plot(ax=ax, zorder=103, lw=10, c='w')
    #         ax.set_xlim(binned.time.value[0], binned.time.value[-1])
    #     else:
    xstep = 0
    xlabels = []
    xlocs = []
    for s in j.sectors:
        lc = j.void[f'clc_{s}'].fold(period=j.results.loc['best', 'overall'])
        xvals = lc.time.value - lc.time.value.min() + xstep
        ax.scatter(xvals, lc.flux, s=1, label=f'Sector(s) {s} Folded')
        xstep = xvals.max()
        if s != j.sectors[-1]:
            ax.axvline(xstep, c='k', ls='-', lw=3, zorder=10)
        xlabels.append(np.nanpercentile(lc.time.value, [25, 50, 75]))
        xlocs.append(np.nanpercentile(xvals, [15, 50, 85]))

        binned = lk.FoldedLightCurve(time=xvals, flux=lc.flux).bin(bins = int(len(lc)/50))
        if s == j.sectors[-1]:
            label = 'Binned LC'
        else:
            label = None
        binned.plot(ax=ax, zorder=104, lw=5, c=cmap[4], label=label)
        binned.plot(ax=ax, zorder=103, lw=10, c='w')
    ax.set_xlim(0, xstep)
    ax.set_xticks(np.array(xlocs).flatten())
    ax.set_xticklabels(np.array(xlabels).flatten().astype(int))

    # else:
    #     lc = j.void[f'clc_all'].fold(period=j.results.loc['best', 'overall'])
    #     lc.scatter(ax=ax, c='k', s=75, label='All Sectors Folded', zorder=1)
    #     binned = lc.bin(bins=int(len(lc)/50))
    #     binned.plot(ax=ax, zorder=104, lw=5, c=cmap[4], label='Binned LC')
    #     binned.plot(ax=ax, zorder=103, lw=10, c='w')
    #     ax.set_xlim(lc.time.value[0], lc.time.value[-1])
    ax.legend(loc='best')
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_title(rf'All Sectors folded on Best Period: {j.results.loc["best", "overall"]:.2f} $\pm$ {j.results.loc["best", "e_overall"]:.2f} d')
    ax.axhline(1.00, lw=5, ls='--', c='k', zorder=100)

def plot(j):
    fig = plt.figure(figsize=(20, 45))
    gs = GridSpec(6,3, figure=fig)

    ax00 = fig.add_subplot(gs[0,0])
    plot_tpf(j, fig, ax00)

    # Plot all LCs
    ax01 = fig.add_subplot(gs[0, 1:])
    plot_lcs(j, fig, ax01)

    # Plot all periodograms
    ax10 = fig.add_subplot(gs[1, :2])
    plot_periodograms(j, fig, ax10)

    # Plot Sector PG Fit
    if np.isfinite(j.results.loc['best','SLS']):
        ax11 = fig.add_subplot(gs[1, 2:], sharey=ax10)
        plot_periodogram_fit(j, fig, ax11)
        ax11.minorticks_on()

    # Wavelet contourfplot
    axw1 = fig.add_subplot(gs[2, :2])
    plot_wavelet_contour(j, fig, axw1)

    # Collapsed Wavelet and fit
    if np.isfinite(j.results.loc['best','SW']):
        axw2 = fig.add_subplot(gs[2, 2:])
        plot_wavelet_fit(j, fig, axw2)
        axw2.minorticks_on()

    # Plot the CACF
    axcf1 = fig.add_subplot(gs[3, :2])
    plot_cacf(j, fig, axcf1)

    # CACF Fit
    if np.isfinite(j.results.loc['best','CACF']):
        axcf2 = fig.add_subplot(gs[3, 2:])
        plot_cacf_fit(j, fig, axcf2)
        axcf2.minorticks_on()


    # Plot the ACF
    acf = fig.add_subplot(gs[4, :2])
    plot_acf(j, fig, acf)

    # Plot the results compared
    res = fig.add_subplot(gs[4, 2:])
    plot_comparison(j, fig, res)

    # Plot the phase folded light curve
    ax2 = fig.add_subplot(gs[5, :])
    plot_fold(j, fig, ax2)

    # Polish
    if j.sectors[0] != 0:
        ax00.minorticks_on()
    ax10.minorticks_on()
    res.grid(axis='y')
    acf.minorticks_on()
    ax2.minorticks_on()
    fig.tight_layout()

    fig.suptitle(f'Gaia ID: {j.gaiaid}', fontsize=30)
    plt.subplots_adjust(top=0.95)


    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.pdf')
    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.png', dpi = 300)
