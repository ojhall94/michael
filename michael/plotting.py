"""
The `plotting` script contains code to plot the results `janet` collects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lightkurve as lk

from utils import *

def plot(j):
    fig = plt.figure(figsize=(20, 20), constrained_layout = True)
    gs = GridSpec(3,3, figure=fig)

    ax00 = self.fig.add_subplot(gs[0, :1])
    ax00.set_title('TPF Frame 0')
    ax00.set_xlabel('X')
    ax00.set_ylabel('Y')

    ax01 = self.fig.add_subplot(gs[0, 1:])
    ax01.set_title('Full TESS LC')

    ax10 = self.fig.add_subplot(gs[1, 2:])
    ax10.set_title('LSP All Sectors')

    ax11 = self.fig.add_subplot(gs[1, :2])

    ax2 = self.fig.add_subplot(gs[2, :])

    ## Plotting
    # Plot Sector 0 TPF
    ax00.imshow(j.void[f'datum_{j.sectors[0]}'].tpf[0])

    # Plot Sector 0 LC
    j.void[f'clc_{j.sectors[0]}'].plot(ax=ax01)

    # Plot Sector All PG
    ax10.plot(j.void['p_all'], j.void['P_all'], lw=10)
    ax10.plot(j.void['p_all'],
            _gaussian_fn(j.void['p_all'], *j.void['popt_all']), ls='--', lw=5)

    # Plot all periodograms
    for s in j.sectors:
        j.void[f'pg_{s}'].plot(ax=ax11, scale='log', view='period',
        label=f'Sector {s}', lw=int(5/(j.results.loc[s, 'f_SLS']+1)))

    ax11.axvline(j.void['popt_all'][0])

    # Plot the phase folded light curve
    j.void['clc_all'].fold(period=j.results.loc['all', 'SLS']).scatter(ax=ax2)
    ax2.set_title(f'All Sectors folded on Period: {j.results.loc["all", "SLS"]:.2f} days')

    # Polish
    fig.tight_layout()
    # self.ax11.get_yaxis().set_visible(False)
    ax10.set_xlabel('Period [d]')
    ax11.set_title('All Periodograms')

    ax00.minorticks_on()
    ax01.minorticks_on()
    ax10.minorticks_on()
    ax11.minorticks_on()
    ax2.minorticks_on()

    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.pdf', rasterized=True)
    plt.savefig(f'{j.output_path}/{j.gaiaid}/output.png', dpi = 300)
