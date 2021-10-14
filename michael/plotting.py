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


def _plot_tpf(j, fig, ax):
    # Plot Sector 0 TPF
    if j.sectors[0] != 0:
        ax.set_title(f'Frame 0 Sector {j.sectors[0]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.imshow(np.log10(j.void[f'datum_{j.sectors[0]}'].tpf[0]), zorder=1)
        pix = np.where(j.void[f'datum_{j.sectors[0]}'].aperture > 0)
        ax.scatter(pix[0], pix[1], edgecolors='w', lw=5, marker=',', facecolors='none', s=600, zorder=2, label='Aperture')
        ax.legend(loc='upper left', fontsize=_label_fontsize)

def _plot_lcs(j, fig, ax):
    ax.set_title(f'Full TESS LC, Sectors: {j.sectors}. Normalised, outliers removed.')
    if len(j.sectors) >= 2:
        if not j.gaps:
            for s in j.sectors:
                j.void[f'clc_{s}'].plot(ax=ax, lw=1)
            ax.set_xlim(j.void[f'clc_all'].time.min().value, j.void[f'clc_all'].time.max().value)
        else:
            xstep = 0
            xlabels = []
            xlocs = []
            for s in j.sectors:
                lc = j.void[f'clc_{s}']
                xvals = lc.time.value - lc.time.value.min() + xstep
                ax.plot(xvals, lc.flux, label=f'Sector {s}', lw=1)
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
    else:
        j.void[f'clc_all'].plot(ax=ax, lw=1, c='k')
        ax.set_xlim(j.void[f'clc_all'].time.min().value, j.void[f'clc_all'].time.max().value)
    ax.set_ylabel('Normalised Flux')

def _plot_periodograms(j, fig, ax):
    best_sls = j.results.loc['best', 's_SLS']

    if not j.gaps:
        j.void[f'pg_all'].plot(ax=ax, view='period', label=f'All Sectors',lw=1, zorder=2, c='k')
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'pg_{s}'].plot(ax=ax, view='period',
            label=f'Sector {s}', lw=1, zorder=2)
    ax.axvline(j.results.loc["best", "SLS"], c=cmap[4], lw=5, ls='--', zorder=1, label=f'P = {j.results.loc["best", "SLS"]:.2f} d')
    ax.set_xlim(j.void[f'pg_{best_sls}'].period.min().value, j.void[f'pg_{best_sls}'].period.max().value)
    ax.set_ylim(0)
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_xscale('log')
    ax.set_title('All Lomb Scargle Periodograms')

def _plot_periodogram_fit(j, fig, ax):
    best_sls = j.results.loc['best', 's_SLS']

    if best_sls == 'all':
        text = 'All Sectors'
    else:
        text = f'Sector {best_sls}'

    ax.get_yaxis().set_visible(False)
    ax.plot(j.void[f'p_{best_sls}'],
            _gaussian_fn(j.void[f'p_{best_sls}'], *j.void[f'popt_{best_sls}']), ls='--', lw=10, c=cmap[5], zorder=2,
            label = rf'$\sigma$ = {j.results.loc["best", "e_SLS"]:.2f} d')
    ax.set_xlim(j.void[f'popt_{best_sls}'][0] - 5*j.void[f'popt_{best_sls}'][1],
                    j.void[f'popt_{best_sls}'][0] + 5*j.void[f'popt_{best_sls}'][1])
    if not j.gaps:
        j.void[f'pg_all'].plot(ax=ax, lw=1, c='k', zorder=0)
    if len(j.sectors) >= 2:
        for s in j.sectors:
            j.void[f'pg_{s}'].plot(ax=ax,lw=1, zorder=0)
    ax.legend(loc='best', fontsize=_label_fontsize)
    ax.set_xlabel('Period [d]')
    ax.set_title(f'Fit to LSP {text}')

def _plot_wavelet_contour(j, fig, ax):
    if not j.gaps:
        ax.contourf(j.void['all_wt'].taus, 1./j.void['all_wt'].nus, j.void['all_wwz'])

        if len(j.sectors) >= 2:
            for s in j.sectors[1:]:
                ax.axvline(j.void[f'clc_{s}'].time.min().value, c='w', ls='-.', lw=3)
            for s in j.sectors:
                ax.text(j.void[f'clc_{s}'].time.min().value+1, (1./j.void[f'{s}_wt'].nus).max() * 0.925, f'S{s}', c='w', weight='bold')
        ax.axhline(j.results.loc['best', 'SW'], ls='--', lw = 3, c='w', label=f'P = {j.results.loc["best", "SW"]:.2f} d')

    else:
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

def _plot_wavelet_fit(j, fig, ax):
    if not j.gaps:
        taus = j.void['all_wt'].taus
        ws = np.sum(j.void['all_wwz'], axis=1)
        ws /= ws.max()
        p = 1/j.void['all_wt'].nus
        ax.plot(p, ws, lw=1, c='k')

        for s in j.sectors:
            time = j.void[f'clc_{s}'].time.value
            sel = (taus >= time.min()) & (taus <= time.max())
            ws = np.sum(j.void['all_wwz'][:, sel], axis=1)
            ws /= ws.max()
            ax.plot(p, ws, lw=1)

    else:
        for s in j.sectors:
            taus = j.void[f'{s}_wt'].taus
            ws = np.sum(j.void[f'{s}_wwz'], axis=1)
            ws /= ws.max()
            p = 1/j.void[f'{s}_wt'].nus
            ax.plot(p, ws, lw=1)

    best_sw = j.results.loc['best', 's_SW']
    if best_sw == 'all':
        text = 'All Sectors'
    else:
        text = f'Sector {best_sw}'

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

def _plot_acf(j, fig, ax):
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
    ax.axvspan(j.results.loc['best', 'overall'] - 2*j.results.loc['best', 'e_overall'],
                j.results.loc['best', 'overall'] + 2*j.results.loc['best', 'e_overall'], color=cmap[7], zorder=1,
                label=r'$2\sigma$', alpha=.5)
    ax.set_title("Autocorrelation Function for all Sectors")
    ax.set_ylabel('Normalised ACF')
    ax.axhline(0.01, label='Detection threshold', c='k', zorder=0)
    ax.legend(loc='upper right')

def _plot_comparison(j, fig, ax):
    ax.set_title('Period Estimates')
    ax.set_ylabel('Period [d]')
    ax.errorbar(2, j.results.loc['all', 'SW'], yerr = j.results.loc['all', 'e_SW'],
                c='k', fmt='o')
    ax.axhline(j.results.loc['all', 'ACF'],  label='ACF', c=cmap[3], ls=':', lw=5,
               zorder =1, alpha=.8)
    # Plot SLS
    if not j.gaps:
        xs = np.linspace(0.8, 1.2, len(j.sectors)+1)
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'SLS'],
                        yerr = j.results.loc[sector, 'e_SLS'], fmt='o', c=colmap[idx])
        ax.errorbar(xs[-1], j.results.loc['all', 'SLS'],
                    yerr = j.results.loc['all', 'e_SLS'],
                    fmt='o', c='k')
    else:
        xs = np.linspace(0.8, 1.2, len(j.sectors))
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'SLS'],
                        yerr = j.results.loc[sector, 'e_SLS'], fmt='o', c=colmap[idx])

    # Plot SW
    if not j.gaps:
        ax.errorbar(2, j.results.loc['all', 'SW'], yerr = j.results.loc['all', 'e_SW'],
                    c='k', fmt='o')
    else:
        xs = np.linspace(1.8, 2.2, len(j.sectors))
        for idx, sector in enumerate(j.sectors):
            ax.errorbar(xs[idx], j.results.loc[sector, 'SW'],
                        yerr = j.results.loc[sector, 'e_SW'], fmt='o', c=colmap[idx])

    labels = ['SLS', 'SW']
    x = [1., 2.]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.axhspan(j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                color=cmap[6], alpha=.5, zorder=0,
               label = 'Best Period')
    if np.any(j.results[['SLS', 'SW', 'ACF']] > 1.9*j.results.loc['best', 'overall']) or\
        np.any(j.results[['SLS', 'SW', 'ACF']] < 0.6*j.results.loc['best', 'overall']):

        ax.axhspan(0.5*j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                    0.5*j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                    color=cmap[7], alpha=.5, zorder=0,
                   label = '2:1:2 Best')
        ax.axhspan(2*j.results.loc['best', 'overall'] - j.results.loc['best', 'e_overall'],
                    2*j.results.loc['best', 'overall'] + j.results.loc['best', 'e_overall'],
                    color=cmap[7], alpha=.5, zorder=0)
    ax.legend(loc='best')
    res = j.results.loc[j.sectors.astype(int), ['SLS','SW']].to_numpy().flatten()
    err = j.results.loc[j.sectors.astype(int), ['e_SLS', 'e_SW']].to_numpy().flatten()
    ax.set_ylim(0.9*np.nanmax(res-err), 1.1*np.nanmax(res+err))

def _plot_fold(j, fig, ax):
    fold = j.void['clc_all'].fold(period=j.results.loc['best', 'overall'])
    if len(j.sectors) >= 2:
        for z, s in enumerate(j.sectors):
            j.void[f'clc_{s}'].fold(period=j.results.loc['best', 'overall']).scatter(s=75, label=f'Sector {s} Folded', ax=ax, zorder=len(j.sectors) - z)
    else:
        fold.scatter(ax=ax, c='k', s=75, label='Folded LC', zorder=1)
    fold.bin(bins=int(len(fold)/50)).plot(ax=ax, zorder=4, lw=5, c=cmap[4], label='Binned LC')
    fold.bin(bins=int(len(fold)/50)).plot(ax=ax, zorder=3, lw=10, c='w')
    ax.set_ylabel('Normalised Flux')
    ax.axhline(1., c='k', zorder=2, ls='-')
    ax.set_xlim(fold.phase.min().value, fold.phase.max().value)
    ax.legend(loc='best', fontsize=_label_fontsize, ncol = int(np.ceil(len(j.sectors)/4)))
    ax.set_title(rf'All Sectors folded on Best Period: {j.results.loc["best", "overall"]:.2f} $\pm$ {j.results.loc["best", "e_overall"]:.2f} d')

def plot(j):
    fig = plt.figure(figsize=(20, 37))
    gs = GridSpec(5,3, figure=fig)

    ax00 = fig.add_subplot(gs[0,0])
    _plot_tpf(j, fig, ax00)

    # Plot all LCs
    ax01 = fig.add_subplot(gs[0, 1:])
    _plot_lcs(j, fig, ax01)

    # Plot all periodograms
    ax10 = fig.add_subplot(gs[1, :2])
    _plot_periodograms(j, fig, ax10)

    # Plot Sector PG Fit
    ax11 = fig.add_subplot(gs[1, 2:], sharey=ax10)
    _plot_periodogram_fit(j, fig, ax11)

    # Wavelet contourfplot
    axw1 = fig.add_subplot(gs[2, :2])
    _plot_wavelet_contour(j, fig, axw1)

    # Collapsed Wavelet and fit
    axw2 = fig.add_subplot(gs[2, 2:])
    _plot_wavelet_fit(j, fig, axw2)

    # Plot the ACF
    acf = fig.add_subplot(gs[3, :2])
    _plot_acf(j, fig, acf)

    # Plot the results compared
    res = fig.add_subplot(gs[3, 2:])
    _plot_comparison(j, fig, res)

    # Plot the phase folded light curve
    ax2 = fig.add_subplot(gs[4, :])
    _plot_fold(j, fig, ax2)

    # Polish
    if j.sectors[0] != 0:
        ax00.minorticks_on()
    # ax01.minorticks_on()
    ax10.minorticks_on()
    ax11.minorticks_on()
    # axw1.minorticks_on()
    axw2.minorticks_on()
    res.grid(axis='y')
    # res.minorticks_on()
    acf.minorticks_on()
    ax2.minorticks_on()
    fig.tight_layout()

    fig.suptitle(f'Gaia ID: {j.gaiaid}', fontsize=30)
    plt.subplots_adjust(top=0.95)


    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.pdf', rasterized=True)
    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.png', dpi = 300)
