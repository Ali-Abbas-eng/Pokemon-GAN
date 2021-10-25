import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import trange
from threading import Thread
import tensorflow as tf
import zipfile


def download_dataset(command: str = "kaggle datasets download -d kvpratama/pokemon-images-dataset --force",
                     path: str = "data",
                     zip_file_name: str = "pokemon-images-dataset.zip"):
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(zip_file_name):
        os.system(command)

    files = os.listdir()
    for file in files:
        if ".zip" in file:
            os.rename(file, zip_file_name)

    with zipfile.ZipFile(zip_file_name, 'r') as dataset_ref:
        dataset_ref.extractall(path)


def custom_plot(plots, rows=2, cols=5, threading=True):
    if threading:
        plot_thread = Thread(target=_custom_plot, args=(plots, rows, cols))
        plot_thread.start()
    else:
        _custom_plot(plots, rows=rows, cols=cols)


def _custom_plot(plots, rows, cols):
    plt.figure(figsize=(50, 50))
    for i in trange(1, plots.shape[0] + 1):
        plt.subplot(rows, cols, i)
        plt.imshow(plots[i - 1])
    plt.show()


def read_images_from_files(path: str = rf"data\pokemon_jpg\pokemon_jpg", plot=False):
    files = [rf"{path}\{file}" for file in os.listdir(path)]
    img = plt.imread(files[0])
    images_data = np.zeros((len(files), img.shape[0], img.shape[1], img.shape[2]))
    for i in trange(len(files)):
        img = plt.imread(files[i])
        images_data[i] += img

    if plot:
        custom_plot(images_data[:10])

    return images_data


def load_data(plot: bool = False):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)
    path = rf"data\pokemon_jpg"
    next_batch = data_generator.flow_from_directory(directory=path,
                                                    target_size=(256, 256),
                                                    batch_size=128,
                                                    class_mode="input")
    if plot:
        custom_plot(next_batch[0][1][:15], rows=3, cols=5)
    return next_batch


if __name__ == "__main__":
    download_dataset()
    data_from_files = read_images_from_files(plot=True)
    load_data(plot=True)