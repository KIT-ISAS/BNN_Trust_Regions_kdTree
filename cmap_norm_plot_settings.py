import matplotlib.colors
import matplotlib.pyplot as plt


def get_anees_plot_settings(vmin=0.3,
                            vmax=1.7,
                            vcenter=1.,
                            cmap_name='seismic',
                            set_under_color='cyan',
                            set_over_color='magenta',):
    # divergent color map seismic
    # set map
    cmap = plt.get_cmap(cmap_name)
    # set two slop normalization
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    if set_under_color is not None:
        cmap.set_under(set_under_color)

    if set_over_color is not None:
        cmap.set_over(set_over_color)

    # use white hashes to indicate out of range values
    # cmap.set_bad(color='white')

    return cmap, norm
