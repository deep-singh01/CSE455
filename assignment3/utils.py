import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import transform
from skimage import io
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm

from segmentation import *

import os

def visualize_mean_color_image(img, segments):

    img = img_as_float(img)
    k = np.max(segments) + 1
    mean_color_img = np.zeros(img.shape)

    for i in range(k):
        mean_color = np.mean(img[segments == i], axis=0)
        mean_color_img[segments == i] = mean_color

    plt.imshow(mean_color_img)
    plt.axis('off')
    plt.show()

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """ Compute a segmentation for an image.

    First a feature vector is extracted from each pixel of an image. Next a
    clustering algorithm is applied to the set of all feature vectors. Two
    pixels are assigned to the same segment if and only if their feature
    vectors are assigned to the same cluster.

    Args:
        img - An array of shape (H, W, C) to segment.
        k - The number of segments into which the image should be split.
        clustering_fn - The method to use for clustering. The function should
            take an array of N points and an integer value k as input and
            output an array of N assignments.
        feature_fn - A function used to extract features from the image.
        scale - (OPTIONAL) parameter giving the scale to which the image
            should be in the range 0 < scale <= 1. Setting this argument to a
            smaller value will increase the speed of the clustering algorithm
            but will cause computed segments to be blockier. This setting is
            usually not necessary for kmeans clustering, but when using HAC
            clustering this parameter will probably need to be set to a value
            less than 1.
    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # Scale down the image for faster computation.
        img = transform.rescale(img, scale)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # Resize segmentation back to the image's original size
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # Resizing results in non-interger values of pixels.
        # Round pixel values to the closest interger
        segments = np.rint(segments).astype(int)

    return segments


def load_dataset(data_dir):
    """
    This function assumes 'gt' directory contains ground truth segmentation
    masks for images in 'imgs' dir. The segmentation mask for image
    'imgs/aaa.jpg' is 'gt/aaa.png'
    """

    imgs = []
    gt_masks = []

    # Load all the images under 'data_dir/imgs' and corresponding
    # segmentation masks under 'data_dir/gt'.
    for fname in sorted(os.listdir(os.path.join(data_dir, 'imgs'))):
        if fname.endswith('.jpg'):
            # Load image
            img = io.imread(os.path.join(data_dir, 'imgs', fname))
            imgs.append(img)

            # Load corresponding gt segmentation mask
            mask_fname = fname[:-4] + '.png'
            gt_mask = io.imread(os.path.join(data_dir, 'gt', mask_fname))
            gt_mask = (gt_mask != 0).astype(int) # Convert to binary mask (0s and 1s)
            gt_masks.append(gt_mask)

    return imgs, gt_masks


def plot_frame(ax, T_local_from_global, label):
    assert T_local_from_global.shape == (4, 4)

    # Get rotation/translation of local origin wrt global frame
    R = T_local_from_global[:3, :3].T
    origin = -R @ T_local_from_global[:3, 3]

    # Draw line for each basis
    for direction, color in zip(R.T, "rgb"):
        ax.quiver(*origin, *direction, color=color, length=0.3, arrow_length_ratio=0.05)

    # Label
    ax.text(origin[0] - 0.1, origin[1], origin[2] + 0.0, "↙" + label, color="black")


def plot_square(ax, vertices):
    return ax.plot3D(vertices[0], vertices[1], vertices[2], "orange",)


def configure_ax(ax):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(elev=20.0, azim=25)


def animate_transformation(
    filename, vertices_wrt_world, camera_from_world_transform, apply_transform,
):
    # Transformation parameters
    d = 1.0

    # Animation parameters
    start_pause = 20
    end_pause = 20

    num_rotation_frames = 20
    num_translation_frames = 20

    # First set up the figure and axes
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    configure_ax(ax)

    # Initial elements
    T_camera_from_world = camera_from_world_transform(d)
    plot_square(ax, vertices=vertices_wrt_world)
    plot_frame(
        ax, T_camera_from_world, label="Camera Frame",
    )
    plot_frame(
        ax, np.eye(4), label="World Frame",
    )

    # Animation function which updates figure data.  This is called sequentially
    def animate(i):
        print(".", end="")
        if i < start_pause:
            return (fig,)
        elif i >= start_pause + num_rotation_frames + num_translation_frames:
            return (fig,)
        else:
            i -= start_pause

        # Disclaimer: this is really inefficient!
        ax.clear()
        configure_ax(ax)
        if i < num_rotation_frames:
            R = expm(logm(T_camera_from_world[:3, :3]) * i / (num_rotation_frames - 1))
            t = np.zeros(3)
        else:
            i -= num_rotation_frames
            R = T_camera_from_world[:3, :3]
            t = i / (num_translation_frames - 1) * T_camera_from_world[:3, 3]

        T_camera_from_world_interp = np.eye(4)
        T_camera_from_world_interp[:3, :3] = R
        T_camera_from_world_interp[:3, 3] = t

        plot_square(
            ax, vertices=apply_transform(T_camera_from_world_interp, vertices_wrt_world)
        )
        plot_frame(
            ax,
            T_camera_from_world @ np.linalg.inv(T_camera_from_world_interp),
            label="Camera Frame",
        )
        plot_frame(
            ax, np.linalg.inv(T_camera_from_world_interp), label="World Frame",
        )

        return (fig,)

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=start_pause + num_rotation_frames + num_translation_frames + end_pause,
        interval=100,
        blit=True,
    )

    anim.save(filename, writer="pillow")
    plt.close()
