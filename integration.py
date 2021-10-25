import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
from tensorflow.keras import losses
from tqdm import trange, tqdm
from discriminator import Discriminator
from generator import Generator
from data_preprocessing import load_data, custom_plot
from tensorflow.keras.optimizers import Adam
from time import sleep
import matplotlib.pyplot as plt

gen_model = Generator()
disc_model = Discriminator((256, 256, 3))

binary_crossentropy_loss = losses.BinaryCrossentropy()
z_dim = 64
batch_size = 128
display_step = 100
lr = .001
num_epochs = 2
dataset = load_data()
current_batch, next_batch = dataset.next()
disc_optimizer = Adam()
gen_optimizer = Adam()
seed = 0
noise_vector_dim = 64


def generate_noise(batch_size, z_dim, seed=0):
    random_generator = tf.random.Generator.from_seed(seed)
    return random_generator.normal(shape=(batch_size, z_dim))


def get_disc_loss(gen, disc, criterion, real, batch_size, z_dim):
    noise = generate_noise(batch_size, z_dim)
    generated_images = gen(noise)

    pred_generated = disc(generated_images)
    true_generated = tf.zeros((batch_size, 1))
    disc_loss_generated = criterion(y_pred=pred_generated, y_true=true_generated)

    pred_real = disc(real)
    true_real = tf.ones((batch_size, 1))
    disc_loss_real = criterion(y_pred=pred_real, y_true=true_real)

    disc_loss = (disc_loss_generated + disc_loss_real) / 2.
    return disc_loss


def get_gen_loss(gen, disc, criterion, batch_size, z_dim):
    noise = generate_noise(batch_size, z_dim)
    generated_images = gen(noise)
    disc_pred = disc(generated_images)
    ture_generated = tf.ones(shape=(batch_size, 1))
    gen_loss = criterion(y_pred=disc_pred, y_true=ture_generated)
    return gen_loss


sleep(.1)
for epoch in range(num_epochs):
    print(rf"Epoch {epoch + 1}\{num_epochs}")
    for i in tqdm(dataset):
        with tf.GradientTape() as disc_tape:
            discriminator_loss = get_disc_loss(gen=gen_model,
                                               disc=disc_model,
                                               criterion=binary_crossentropy_loss,
                                               real=current_batch,
                                               batch_size=batch_size,
                                               z_dim=noise_vector_dim)
        disc_gradients = disc_tape.gradient(discriminator_loss, disc_model.trainable_weights)
        disc_optimizer.apply_gradients(zip(disc_gradients, disc_model.trainable_weights))

        with tf.GradientTape() as generator_tape:
            generator_loss = get_gen_loss(gen=gen_model,
                                          disc=disc_model,
                                          criterion=binary_crossentropy_loss,
                                          batch_size=batch_size,
                                          z_dim=noise_vector_dim)
            gen_gradients = generator_tape.gradient(generator_loss, gen_model.trainable_weights)
            gen_optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_weights))
