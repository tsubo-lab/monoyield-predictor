# User-specified plotting style; call set_style() at the top of every reproduction script.
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def set_style(font_size=6, palette='tab10'):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", palette=sns.color_palette(palette), rc=custom_params)
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2
    plt.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
        "figure.titlesize": font_size,
        "font.family": "Arial",
        'mathtext.default': 'default',
        'mathtext.fontset': 'stix',
    })
