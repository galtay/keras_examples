import numpy as np
import matplotlib.pyplot as plt

def network_architecture(model, xmin=0.0, xmax=1.0, ymin=0.0, ymax=0.6):
    """Plot a feed forward fully connected network architecture."""

    ymid = (ymin + ymax) * 0.5
    nlyrs = len(model.layers)

    # we add + 1 so we have a potential bias node for each layer
    nds_per_lyr = [lyr.output_shape[-1] + 1 for lyr in model.layers]
    max_nds = max(nds_per_lyr)
    dx = xmax / nlyrs
    dy = ymax / max_nds


    # set positions for nodes (list of lists of tuples: layer->node->coords)
    #------------------------------------------------------------------------
    xpos = (np.arange(nlyrs) + 0.5) * dx
    nodes_pos = []
    for ilyr, lyr in enumerate(model.layers):
        n_nodes = nds_per_lyr[ilyr]
        ypos = (np.arange(n_nodes) + 0.5) * dy
        yadj = ymid - (ypos.max() + ypos.min()) * 0.5
        ypos += yadj
        tmp = []
        for inode in range(n_nodes):
            tmp.append((xpos[ilyr], ypos[inode]))
        nodes_pos.append(tmp)

    fig, ax = plt.subplots(figsize=(10*xmax, 10*ymax))

    # draw lines (loop over pairs of layers)
    #------------------------------------------------------------------------
    for lyr_left, lyr_right in zip(nodes_pos[:-1], nodes_pos[1:]):

        # draw bias weights
        for node_left in lyr_left[-1:]:
            for node_right in lyr_right[:-1]:
                xl=node_left[0]; xr=node_right[0]
                yl=node_left[1]; yr=node_right[1]
                plt.plot([xl,xr], [yl,yr], color='black', lw=1.0, ls='--', zorder=1)

        # draw non-bias weights
        for node_left in lyr_left[:-1]:
            for node_right in lyr_right[:-1]:
                xl=node_left[0]; xr=node_right[0]
                yl=node_left[1]; yr=node_right[1]
                plt.plot([xl,xr], [yl,yr], color='black', lw=1.0, ls='-', zorder=1)

    # draw nodes
    #------------------------------------------------------------------------
    for ilyr, lyr in enumerate(nodes_pos):
        # draw bias nodes (all layers but last)
        if ilyr < nlyrs - 1:
            for pos in lyr[-1:]:
                x=pos[0]
                y=pos[1]
                circle = plt.Circle((x,y), radius=0.035, color='red')
                ax.add_artist(circle)
        # draw regular nodes
        for pos in lyr[:-1]:
            x=pos[0]
            y=pos[1]
            circle = plt.Circle((x,y), radius=0.035, color='blue')
            ax.add_artist(circle)

    # final adjustments
    #------------------------------------------------------------------------
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
