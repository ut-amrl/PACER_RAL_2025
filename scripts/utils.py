import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


PATCH_SIZE = 32
def patches_to_tensor(patches_list):
    # make an empty 9xPATCH_SIZE * 2xPATCH_SIZE np array
    arr = np.empty((9, PATCH_SIZE * 2, PATCH_SIZE))

    for i in range(3):
        img1 = patches_list[i*2]
        img2 = patches_list[i*2+1]
        
        # concatenate these images up and down
        img = np.concatenate((img1, img2), axis=0)
        img = img_to_tensor_format(img, new_size=(PATCH_SIZE, PATCH_SIZE * 2), to_rgb=True)
        arr[i*3:i*3+3] = img
    return torch.tensor(arr).unsqueeze(0).float()


def img_to_tensor_format(img, new_size = None, to_rgb = True, transpose = True, normalize = True):
    if new_size is not None:
        # resize img to half the size
        img = cv2.resize(img, new_size)
    if to_rgb:
        # bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        # convert to np array and divide by 255
        img = np.array(img) / 255.0
    if transpose:
        # change to c, h, w
        img = np.transpose(img, (2, 0, 1))
    return img


def resize_image(shape, img):
    # Assuming img is your image and pref_costmap is your target size
    target_height, target_width = shape[0], shape[1]
    original_height, original_width = img.shape[0], img.shape[1]

    # Calculate the scaling factors
    scale_width = target_width / original_width
    scale_height = target_height / original_height

    # Use the smaller scaling factor
    scale = max(scale_width, scale_height)

    # Calculate the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    img = cv2.resize(img, (new_width, new_height))
    return img


def viz_contexts(contexts):
    B = contexts.shape[0]
    fig, axs = plt.subplots(B, figsize=(3, B))
    contexts = torch.tensor(contexts)
    contexts = contexts.reshape(B, 3, 3, 64, 32)  # reshape to (batch_size, 3, 3, 64, 32)

    for i in range(B):
        context = contexts[i]
        context_grid = make_grid(context, normalize=True, nrow=3)
        context_grid = context_grid.permute(1, 2, 0).numpy()
        axs[i].imshow(context_grid)
        axs[i].set_title("Context {}".format(i))
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig("context_viz.png")

def viz_icl_dataset_batch(batch):
    # Assume batch_size is the number of examples you want to plot

    contexts = batch['context']
    bev_costmaps = batch['bev_costmap']
    bev_images = batch['bev_image']
    prefs = batch['pref']

    batch_size = contexts.shape[0]

    fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    for i in range(batch_size):
        # context is a 9 x 64 x 32 tensor. take every 3 channels as an image
        context = torch.tensor(contexts[i])
        context = context.reshape(3, 3, 64, 32)  # reshape to (3, 3, 64, 32)

        context_grid = make_grid(context, normalize=True, nrow=3)
        bev_image_grid = make_grid(torch.tensor(bev_images[i]), nrow=1)
        bev_costmap_grid = make_grid(torch.tensor(bev_costmaps[i]), nrow=1)

        # convert to numpy for visualization
        context_grid = context_grid.permute(1, 2, 0).numpy()
        bev_image_grid = bev_image_grid.permute(1, 2, 0).numpy()
        bev_costmap_grid = bev_costmap_grid.permute(1, 2, 0).numpy()

        # plot context_grid, bev_image_grid, bev_costmap_grid in a row
        axs[i, 0].imshow(context_grid)
        axs[i, 0].set_title("context", pad=40)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(bev_image_grid)
        axs[i, 1].set_title("BEV", pad=40)
        # axs[i, 1].set_title(f"Preference Ordering: {pref_ordering[i1]}", pad=40)
        axs[i, 1].axis('off')
        axs[i, 2].imshow(bev_costmap_grid, vmin=0, vmax=1)
        axs[i, 2].set_title("Target Costmap", pad=40)
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("icl_dataset_batch.png")
    plt.close('all')
