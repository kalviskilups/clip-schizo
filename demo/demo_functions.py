"""This file contains the functions that are used in the demo."""

import sys
sys.path.append('..')
from src import imageops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import streamlit as st

colorsl = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 255, 255], [128, 128, 128], [128, 0, 0], [0, 128, 0], [0, 0, 128]]
colors = np.array(colorsl)
colorsl = [[c / 255 for c in color] for color in colorsl]

def array_to_rgb(array):
    n, y, x = array.shape
    max_indices = np.argmax(array, axis=0)
    rgb_image = np.zeros((y, x, 3), dtype=np.uint8)
    for i in range(n):
        mask = max_indices == i
        rgb_image[mask] = colors[i]

    return rgb_image

def run_image(smodel, path, labels, multilabel = False, aggregation = 'max', rough_labels = None, web = False):
    img = imageops.open_image(path)
    if multilabel == False:
        hmaps = smodel.forward(img, labels)
    else:
        hmaps = smodel.forward_multilabel(img, labels, aggregation = aggregation)

    fig = plt.figure(figsize=(28, 8))

    width = ((len(labels) + 2 + 1) // 2) * 2
    gs = gridspec.GridSpec(2, width, width_ratios=[1, 0.05]*(width // 2))

    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, aspect = 'auto')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(array_to_rgb(hmaps), aspect = 'auto')
    ax1.set_xticks([])
    ax1.set_yticks([])

    if multilabel:
        if rough_labels is None:
            labels_s = [l[0] for l in labels]
        else:
            labels_s = rough_labels
    else:
        labels_s = labels

    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colorsl[:len(labels_s)], labels_s)]
    ax1.legend(handles=legend_patches)

    # Individual heatmaps and their colorbars
    for idx, l in enumerate(labels):
        ax = fig.add_subplot(gs[(idx * 2 + 4) // width, (idx*2 + 4) % width])
        cax = fig.add_subplot(gs[(idx * 2 + 5) // width, (idx*2 + 5) % width])
        pos = cax.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0, pos.width, pos.height]
        cax.set_position(new_pos)
        ax.set_xticks([])
        ax.set_yticks([])

        im = ax.imshow(hmaps[idx], aspect = 'auto', vmin = 0, vmax = 1)
        plt.colorbar(im, cax=cax, pad = 0)
        ax.set_title(labels_s[idx])

    if web:
        st.pyplot(fig)
    else:
        plt.savefig(path[:-4] + 'output' + '.png', bbox_inches = 'tight')