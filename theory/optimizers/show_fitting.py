import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from theory.models.constants import G_0_muS, e, h
from utilities.corporate_design_colors_v4 import colors, cmap

NDArray64 = NDArray[np.float64]
DictType = dict[str, float | NDArray[np.float64 | np.bool] | None]


def show_fitting(solution: DictType, num: int = 0):

    V_mV: NDArray64 = np.array(solution["V_mV"], dtype=np.float64)
    V_nan_0_mV = np.where(V_mV == 0.0, np.nan, V_mV)
    m_pos = V_mV > 0
    m_neg = V_mV < 0

    I_exp: NDArray64 = np.array(solution["I_exp_nA"], dtype=np.float64)
    I_ini: NDArray64 = np.array(solution["I_ini_nA"], dtype=np.float64)
    I_fit: NDArray64 = np.array(solution["I_fit_nA"], dtype=np.float64)

    popt: NDArray64 = np.array(solution["popt"], dtype=np.float64)
    perr: NDArray64 = np.array(solution["perr"], dtype=np.float64)

    g_exp = I_exp / V_nan_0_mV / G_0_muS
    g_ini = I_ini / V_nan_0_mV / G_0_muS
    g_fit = I_fit / V_nan_0_mV / G_0_muS

    G_exp = np.gradient(I_exp, V_mV) / G_0_muS
    G_ini = np.gradient(I_ini, V_mV) / G_0_muS
    G_fit = np.gradient(I_fit, V_mV) / G_0_muS

    V_mV_pos = V_mV[m_pos]
    V_mV_neg = -V_mV[m_neg]

    I_exp_pos = I_exp[m_pos]
    I_exp_neg = -I_exp[m_neg]
    I_ini_pos = I_ini[m_pos]
    I_ini_neg = -I_ini[m_neg]
    I_fit_pos = I_fit[m_pos]
    I_fit_neg = -I_fit[m_neg]

    g_exp_pos = g_exp[m_pos]
    g_exp_neg = g_exp[m_neg]
    g_fit_pos = g_fit[m_pos]
    g_fit_neg = g_fit[m_neg]
    g_ini_pos = g_ini[m_pos]
    g_ini_neg = g_ini[m_neg]

    ug_fit_pos = +(g_fit_pos**2) - g_exp_pos**2
    ug_fit_neg = -(g_fit_neg**2) + g_exp_neg**2
    ug_ini_pos = +(g_ini_pos**2) - g_exp_pos**2
    ug_ini_neg = -(g_ini_neg**2) + g_exp_neg**2

    G_exp_pos = G_exp[m_pos]
    G_exp_neg = G_exp[m_neg]
    G_fit_pos = G_fit[m_pos]
    G_fit_neg = G_fit[m_neg]
    G_ini_pos = G_ini[m_pos]
    G_ini_neg = G_ini[m_neg]

    uG_fit_pos = +(G_fit_pos**2) - G_exp_pos**2
    uG_fit_neg = -(G_fit_neg**2) + G_exp_neg**2
    uG_ini_pos = +(G_ini_pos**2) - G_exp_pos**2
    uG_ini_neg = -(G_ini_neg**2) + G_exp_neg**2

    Delta_mV = solution["Delta_mV"]
    tau = solution["tau"]

    plt.close(num)
    fig, axs = plt.subplots(
        num=num, nrows=5, sharex=True, height_ratios=(2, 1, 1, 1, 1), figsize=(6, 9)
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    def VtoE(V_mV):
        return V_mV / Delta_mV

    def EtoV(E_meV):
        return E_meV * Delta_mV

    def G_G_0_to_G_arbu(x):
        return x / tau

    def G_arbu_to_G_G_0(x):
        return x * tau

    def I_nA_to_I_arbu(x):
        return x / (1e-9 * 2 * e / h * Delta_mV * 1e-3)

    def I_arbu_to_I_nA(x):
        return x * (1e-9 * 2 * e / h * Delta_mV * 1e-3)

    axs_2 = []
    for ax in axs:
        ax.tick_params(
            direction="in",  # Ticks nach innen
            top=False,  # obere Ticks ein
            bottom=True,  # untere Ticks ein
            left=True,  # linke Ticks ein
            right=False,  # rechte Ticks ein
            which="both",  # sowohl Major- als auch Minor-Ticks
        )
        ax.grid()
        ax.tick_params(labelbottom=False)
        ax_2 = ax.secondary_xaxis("top", functions=(VtoE, EtoV))
        ax_2.tick_params(direction="in", top=True)
        ax_2.tick_params(labeltop=False)
        axs_2.append(ax_2)
    axs_2[0].tick_params(labeltop=True)
    axs[-1].tick_params(labelbottom=True)

    (ax_I, ax_g, ax_G, ax_ug, ax_uG) = axs
    (ax_I_2, ax_g_2, ax_G_2, ax_ug_2, ax_uG_2) = axs_2
    axs_s = []
    for ax in [ax_ug, ax_uG]:
        ax_s = ax.twinx()
        ax_s.tick_params(axis="y", labelcolor="grey", direction="in", color="grey")
        ax_s.set_ylabel("$\\sigma$ (arb. u.)", color="grey")
        ax_s.set_ylim((-0.1, 1.1))
        axs_s.append(ax_s)

    axs_ = []
    for ax in [ax_g, ax_G]:
        ax_ = ax.secondary_yaxis("right", functions=(G_G_0_to_G_arbu, G_arbu_to_G_G_0))
        ax_.tick_params(direction="in", right=True)
        ax_.tick_params(labelleft=False)
        axs_.append(ax_)

    ax_I_ = ax_I.secondary_yaxis("right", functions=(I_nA_to_I_arbu, I_arbu_to_I_nA))
    ax_I_.tick_params(direction="in", right=True)
    ax_I_.tick_params(labelleft=False)

    ax_I_2.set_xlabel("$E$ ($\\Delta$)")
    ax_uG.set_xlabel("$V$ (mV)")

    ax_I.set_ylabel("$I$ (nA)")
    ax_I_.set_ylabel("$I$ ($2e/h \\cdot \\Delta$)")
    ax_g.set_ylabel("$I/V$ ($G_0$)")
    axs_[0].set_ylabel("$I/V$ ($G$)")
    ax_G.set_ylabel("d$I/$d$V$ ($G_0$)")
    axs_[1].set_ylabel("d$I/$d$V$ ($G$)")
    ax_ug.set_ylabel("$u_{I/V}$ ($G_0^2$)")
    ax_uG.set_ylabel("$u_{\\mathrm{d}I/\\mathrm{d}V}$ ($G_0^2$)")

    ax_I.plot(
        V_mV_pos,
        I_exp_pos,
        ".",
        label="$I_\\mathrm{exp}^\\rightarrow$",
        color=colors(3),
        ms=2,
        zorder=13,
    )
    ax_I.plot(
        V_mV_neg,
        I_exp_neg,
        ".",
        label="$I_\\mathrm{exp}^\\leftarrow$",
        color=colors(3, 0.3),
        ms=2,
        zorder=13,
    )
    ax_I.plot(V_mV_pos, I_fit_pos, label="$\\mathrm{fit}$", color=colors(0), zorder=12)
    ax_I.plot(V_mV_neg, I_fit_neg, color=colors(0), zorder=12)
    ax_I.plot(V_mV_pos, I_ini_pos, label="$\\mathrm{ini}$", color=colors(4), zorder=11)
    ax_I.plot(V_mV_neg, I_ini_neg, color=colors(4), zorder=11)
    ax_I.legend()

    ax_g.plot(V_mV_pos, g_exp_pos, ".", color=colors(3), ms=2, zorder=13)
    ax_g.plot(V_mV_neg, g_exp_neg, ".", color=colors(3, 0.3), ms=2, zorder=13)
    ax_g.plot(V_mV_pos, g_fit_pos, color=colors(0), zorder=12)
    ax_g.plot(V_mV_neg, g_fit_neg, color=colors(0), zorder=12)
    ax_g.plot(V_mV_pos, g_ini_pos, color=colors(4), zorder=11)
    ax_g.plot(V_mV_neg, g_ini_neg, color=colors(4), zorder=11)

    ax_G.plot(V_mV_pos, G_exp_pos, ".", color=colors(3), ms=2, zorder=13)
    ax_G.plot(V_mV_neg, G_exp_neg, ".", color=colors(3, 0.3), ms=2, zorder=13)
    ax_G.plot(V_mV_pos, G_fit_pos, color=colors(0), zorder=12)
    ax_G.plot(V_mV_neg, G_fit_neg, color=colors(0), zorder=12)
    ax_G.plot(V_mV_pos, G_ini_pos, color=colors(4), zorder=11)
    ax_G.plot(V_mV_neg, G_ini_neg, color=colors(4), zorder=11)

    ax_ug.plot(
        V_mV_pos, np.full_like(V_mV_pos, 0.0), ".", color=colors(3), ms=2, zorder=13
    )
    ax_ug.plot(V_mV_pos, ug_fit_pos, color=colors(0), zorder=12)
    ax_ug.plot(V_mV_neg, ug_fit_neg, color=colors(0, 0.3), zorder=12)
    ax_ug.plot(V_mV_pos, ug_ini_pos, color=colors(4), zorder=11)
    ax_ug.plot(V_mV_neg, ug_ini_neg, color=colors(4, 0.3), zorder=11)

    ax_uG.plot(
        V_mV_pos, np.full_like(V_mV_pos, 0.0), ".", color=colors(3), ms=2, zorder=13
    )
    ax_uG.plot(V_mV_pos, uG_fit_pos, color=colors(0), zorder=12)
    ax_uG.plot(V_mV_neg, uG_fit_neg, color=colors(0, 0.3), zorder=12)
    ax_uG.plot(V_mV_pos, uG_ini_pos, color=colors(4), zorder=11)
    ax_uG.plot(V_mV_neg, uG_ini_neg, color=colors(4, 0.3), zorder=11)

    axs_s[0].plot(V_mV_pos, np.full_like(V_mV_pos, 1.0), color="grey")
    axs_s[1].plot(V_mV_pos, np.full_like(V_mV_pos, 0.0), color="grey")

    # add text box for the statistics
    stats = ""
    stats += f"$G={popt[0]:.4f}$"
    stats += f"$\\,({int(np.round(perr[0]*1e4))})$" if perr[0] != 0 else ""
    stats += "$\\,G_0$\n"
    stats += f"$T={popt[1]*1e3:.1f}$"
    stats += f"$\\,({int(np.round(perr[1]*1e4))})$" if perr[1] != 0 else ""
    stats += "$\\,$mK\n"
    stats += f"$\\Delta={popt[2]*1e3:.1f}$"
    stats += f"$\\,({int(np.round(perr[2]*1e4))})$" if perr[2] != 0 else ""
    stats += "$\\,$µeV\n"
    stats += f"$\\Gamma={popt[3]*1e3:.2f}$"
    stats += f"$\\,({int(np.round(perr[3]*1e5))})$" if perr[3] != 0 else ""
    stats += "$\\,$µeV"
    bbox = dict(boxstyle="round", fc="lightgrey", ec="grey", alpha=0.5)
    ax_I.text(
        0.05,
        0.6,
        stats,
        fontsize=9,
        bbox=bbox,
        transform=ax_I.transAxes,
        horizontalalignment="left",
    )

    fig.tight_layout()
