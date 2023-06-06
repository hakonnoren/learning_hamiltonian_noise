import matplotlib as mpl


def set_plot_params():
    golden_ratio = (5**.5 - 1) / 2
    params = {
        # Use the golden ratio to make plots aesthetically pleasing
        'figure.figsize': [5, 5*golden_ratio],
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document, titles slightly larger
        # In the end, most plots will anyhow be shrunk to fit onto a US Letter / DIN A4 page
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
    mpl.rcParams.update(params)


