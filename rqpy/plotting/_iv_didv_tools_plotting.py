import numpy as np
import rqpy as rp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


__all__ = ["_make_iv_noiseplots",
           "_plot_energy_res_vs_bias",
           "_plot_rload_rn_qetbias",
           "_plot_didv_bias",
          ]


def _make_iv_noiseplots(IVanalysisOBJ, lgcsave=False):
    """
    Helper function to plot average noise/didv traces in time domain, as well as 
    corresponding noise PSDs, for all QET bias points in IV/dIdV sweep.

    Parameters
    ----------
    IVanalysisOBJ : rqpy.IVanalysis
         The IV analysis object that contains the data to use for plotting.
    lgcsave : bool, optional
        If True, all the plots will be saved in the a folder avetrace_noise/ within
        the user specified directory.

    Returns
    -------
    None

    """

    for (noiseind, noiserow), (didvind, didvrow) in zip(IVanalysisOBJ.df[IVanalysisOBJ.noiseinds].iterrows(),
                                                        IVanalysisOBJ.df[IVanalysisOBJ.didvinds].iterrows()):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        t = np.arange(0,len(noiserow.avgtrace))/noiserow.fs
        tdidv = np.arange(0, len(didvrow.avgtrace))/noiserow.fs
        axes[0].set_title(f"{noiserow.seriesnum} Avg Trace, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
        axes[0].plot(t*1e6, noiserow.avgtrace * 1e6, label=f"{self.chname} Noise", alpha=0.5)
        axes[0].plot(tdidv*1e6, didvrow.avgtrace * 1e6, label=f"{self.chname} dIdV", alpha=0.5)
        axes[0].grid(which="major")
        axes[0].grid(which="minor", linestyle="dotted", alpha=0.5)
        axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        axes[0].set_ylabel("Current [μA]", fontsize = 14)
        axes[0].set_xlabel("Time [μs]", fontsize = 14)
        axes[0].legend()

        axes[1].loglog(noiserow.f, noiserow.psd**0.5 * 1e12, label=f"{self.chname} PSD")
        axes[1].set_title(f"{noiserow.seriesnum} PSD, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
        axes[1].grid(which="major")
        axes[1].grid(which="minor", linestyle="dotted", alpha=0.5)
        axes[1].set_ylim(1, 1e3)
        axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        axes[1].set_ylabel(r"PSD [pA/$\sqrt{\mathrm{Hz}}$]", fontsize = 14)
        axes[1].set_xlabel("Frequency [Hz]", fontsize = 14)
        axes[1].legend()

        plt.tight_layout()
        if lgcsave:
            if not savepath.endswith('/'):
                savepath += '/'
            fullpath = f'{IVanalysisOBJ.figsavepath}avetrace_noise/'
            if not os.path.isdir(fullpath):
                os.makedirs(fullpath)

            plt.savefig(fullpath + f'{noiserow.qetbias*1e6:.2f}_didvnoise.png')
        plt.show()

def _plot_rload_rn_qetbias(IVanalysisOBJ, lgcsave, xlims_rl, ylims_rl, xlims_rn, ylims_rn):
    """
    Helper function to plot rload and rnormal as a function of
    QETbias from the didv fits of SC and Normal data for IVanalysis object.

    Parameters
    ----------
    IVanalysisOBJ : rqpy.IVanalysis
        The IV analysis object that contains the data to use for plotting.
    lgcsave : bool, optional
        If True, all the plots will be saved 
    xlims_rl : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim() for the  rload plot
    ylims_rl : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim() for the rload plot
    xlims_rn : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim() for the  rtot plot
    ylims_rn : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim() for the rtot plot

    Returns
    -------
    None

    """

    fig, axes = plt.subplots(1,2, figsize = (16,6))
    fig.suptitle("Rload and Rtot from dIdV Fits", fontsize = 18)

    if xlims_rl is not None:
        axes[0].set_xlim(xlims_rl)
    if ylims_rl is not None:
        axes[0].set_ylim(ylims_rl)
    if xlims_rn is not None:
        axes[1].set_xlim(xlims_rn)
    if ylims_rn is not None:
        axes[1].set_ylim(ylis_rn)

    axes[0].errorbar(IVanalysisOBJ.vb[0,0,IVanalysisOBJ.scinds]*1e6,
                     np.array(IVanalysisOBJ.rload_list)*1e3, 
                     yerr = IVanalysisOBJ.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
    axes[0].grid(True, linestyle = 'dashed')
    axes[0].set_title('Rload vs Vbias', fontsize = 14)
    axes[0].set_ylabel(r'$R_ℓ$ [mΩ]', fontsize = 14)
    axes[0].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
    axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")

    axes[1].errorbar(IVanalysisOBJ.vb[0,0,IVanalysisOBJ.norminds]*1e6,
                     np.array(IVanalysisOBJ.rtot_list)*1e3, 
                     yerr = IVanalysisOBJ.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
    axes[1].grid(True, linestyle = 'dashed')
    axes[1].set_title('Rtotal vs Vbias', fontsize = 14)
    axes[1].set_ylabel(r'$R_{N} + R_ℓ$ [mΩ]', fontsize = 14)
    axes[1].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
    axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")

    plt.tight_layout()
    if lgcsave:
        plt.savefig(IVanalysisOBJ.figsavepath + 'rload_rtot_variation.png')


def _plot_energy_res_vs_bias(r0s, 
                             energy_res, 
                             qets, 
                             taus,
                             xlims=None, 
                             ylims=None, 
                             lgctau=False, 
                             lgcoptimum=False,
                             figsavepath='', 
                             lgcsave=False,
                             energyscale=None,):
    """
    Helper function for the IVanalysis class to plot the expected energy resolution as 
    a function of QET bias and TES resistance.

    Parameters
    ----------
    r0s : ndarray
        Array of r0 values (in Ohms)
    energy_res : ndarray
        Array of expected energy resolutions (in eV)
    qets : ndarray
        Array of QET bias values (in Amps)
    taus : ndarray
        Array of tau minus values (in seconds)
    xlims : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim()
    lgctau : bool, optional
        If True, tau_minus is plotted as 
        function of R0 and QETbias
    lgcoptimum : bool, optional
        If True, the optimum energy res
        (and tau_minus if lgctau=True)
    figsavepath : str, optional
        Directory to save the figure
    lgcsave : bool, optional
        If true, the figure is saved
    energyscale : char, NoneType, optional
        The metric prefix for how the energy
        resolution should be scaled. Defaults
        to None, which will be base units [eV].
        Can be: 'n->nano, u->micro, m->milla,
        k->kilo, M-Mega, G-Giga'

    Returns
    -------
    None

    """

    metric_prefixes = {'n' : 1e9, 
                       'u' : 1e6,
                       'm' : 1e3, 
                       'k' : 1e-3,
                       'M' : 1e-6, 
                       'G' : 1e-9}
    if energyscale is None:
        scale = 1
        energyscale = ''
    elif energyscale not in metric_prefixes:
        raise ValueError(f'energyscale must be one of {metric_prefixes.keys()}')
    else:
        scale = metric_prefixes[energyscale]
    if energyscale == 'u':
        energyscale = r'$\mu$'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

    if xlims is None:
        xlims = (min(r0s*1e3), max(r0s*1e3))
    if ylims is None:
        ylims = (min(energy_res*scale), max(energy_res*scale))
    crangey = rp.inrange(energy_res, ylims[0], ylims[1])
    crangex = rp.inrange(r0s*1e3, xlims[0], xlims[1])

    r0s = r0s[crangey & crangex]*1e3
    energy_res = energy_res[crangey & crangex]*scale
    qets = (qets[crangey & crangex]*1e6).round().astype(int)
    taus = taus[crangey & crangex]*1e6

    ax.plot(r0s, energy_res, linestyle = ' ', marker = '.', ms = 10, c='g')
    ax.plot(r0s, energy_res, linestyle = '-', marker = ' ', linewidth = 3, alpha = .5, c='g')
    ax.grid(True, which = 'both', linestyle = '--')
    ax.set_xlabel('$R_0$ [mΩ]')
    ax.set_ylabel(r'$σ_E$'+f' [{energyscale}eV]', color='g')
    ax.tick_params('y', colors='g')
    ax.tick_params(which="both", direction="in", right=True, top=True)

    if lgcoptimum:
        plte = ax.axvline(r0s[np.argmin(energy_res)], linestyle = '--', color='g', 
                    alpha=0.5, label=r'Min $\sigma_E$: '+f'{np.min(energy_res):.3f} [{energyscale}eV]')

    ax2 = ax.twiny()
    ax2.spines['bottom'].set_position(('outward', 36))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom') 
    ax2.set_xticks(r0s)
    ax2.set_xticklabels(qets)
    ax2.set_xlabel(r'QET bias [$\mu$A]')



    if lgctau:
        ax3 = ax.twinx()
        ax3.plot(r0s, taus, linestyle = ' ', marker = '.', ms = 10, c='b')
        ax3.plot(r0s, taus, linestyle = '-', marker = ' ', linewidth = 3, alpha = .5, c='b')
        ax3.tick_params(which="both", direction="in", right=True, top=True)
        ax3.tick_params('y', colors = 'b')
        ax3.set_ylabel(r'$\tau_{-} [μs]$', color = 'b')

        if lgcoptimum:
            plttau = ax3.axvline(r0s[np.argmin(taus)], linestyle = '--', color='b', 
                        alpha=0.5, label=r'Min $\tau_{-}$: '+f'{np.min(taus):.3f} [μs]')
    if xlims is not None:
        ax.set_xlim(xlims)
        ax2.set_xlim(xlims)
        if lgctau:
            ax3.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_title('Expected Energy Resolution vs QET bias and $R_0$')
    if lgcoptimum:
        ax.legend()
        if lgctau:
            ax.legend(loc='upper center', handles=[plte, plttau])

    if lgcsave:
        plt.savefig(f'{figsavepath}energy_res_vs_bias.png')
        
def _plot_didv_bias(data, xlims=(-.15,0.025), ylims=(0,.08),
                   cmap='magma'):
    """
    Helper function to plot the real vs imaginary
    part of the didv for different QET bias values
    for an IVanalysis object
    
    Parameters
    ----------
    data : IVanalysis object
        The IVanalysis object with the didv fits
        already done
    xlims : tuple, optional
        The xlimits of the plot
    ylims : tuple, optional
        The ylimits of the plot
    cmap : str, optional
        The colormap to use for the 
        plot. 
        
    Returns
    -------
    fig, ax : matplotlib fig and axes objects
    """
    
    fig,ax=plt.subplots(figsize=(10,6))
    ax.set_xlabel('Re($dI/dV$) ($\Omega^{-1}$)')
    ax.set_ylabel('Im($dI/dV$) ($\Omega^{-1}$)')

    ax.set_title("Real and Imaginary Part of dIdV")
    ax.tick_params(which='both',direction='in',right=True,top=True)
    ax.grid(which='major')
    ax.grid(which='minor',linestyle='dotted',alpha=0.3)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    qets = np.abs(data.df.loc[data.didvinds, 'qetbias'].iloc[data.traninds].values)*1e6

    normalize = mcolors.Normalize(vmin=min(qets), vmax=max(qets))
    colormap = plt.get_cmap(cmap)
    ax.grid(True, linestyle='--')

    for ind in (data.traninds):
        ii = ind-data.traninds[0]
        row = data.df[data.didvinds].iloc[ind]
        didv = row.didvobj2
        goodinds=np.abs(didv.didvmean/didv.didvstd) > 2.0 
        fitinds = didv.freq>0
        plotinds= np.logical_and(fitinds, goodinds)
        ax.plot(np.real(didv.didvmean)[plotinds],np.imag(didv.didvmean)[plotinds], linestyle=' ',
                marker ='.', alpha = 0.9, ms=10,
                  c=colormap(normalize(qets[ii])))

        ax.plot(np.real(didv.didvfit_freqdomain)[fitinds],np.imag(didv.didvfit_freqdomain)[fitinds],
                    c=colormap(normalize(qets[ii])) )

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(qets[:-1])
    cbar = plt.colorbar(scalarmappaple)
    cbar.set_label('QET Bias [μA]', labelpad = 3) 
    
    return fig, ax
