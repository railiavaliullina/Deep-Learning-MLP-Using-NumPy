import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title=''):
    if len(image.shape) > 2:  # rgb
        image = image.transpose((1, 2, 0))
    plt.imshow(image)
    plt.title(title)
    plt.show()


def clamp(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def show_batch(batch, labels=None, title='', show=False, save=False, path=None, grid_size=None, set_labels=True):
    """
    Создание сетки с изображениями в батче
    """
    images, labels = batch
    n_rows, n_cols = grid_size
    f, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.flatten()
    for i, (img, ax) in enumerate(zip(images, axs)):
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        img = clamp(img)
        if i < len(images):
            ax.imshow(img)
            if set_labels:
                ax.set_title(f'label: {labels[i]}', fontsize=8)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        # ax.set_aspect('equal')
    # f.delaxes(axs[-1])
    if save:
        plt.savefig(path)
    if show:
        plt.show()
