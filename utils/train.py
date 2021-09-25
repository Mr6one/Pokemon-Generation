import numpy as np
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from collections import defaultdict
from IPython.display import clear_output

import torch
from torch.distributions.categorical import Categorical
from models.biggan import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_dcgan_history(history, num_epochs):
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, history['discriminator_loss'], label='discriminator loss')
    plt.plot(np.arange(num_epochs) + 1, history['generator_loss'], label='generator loss')
    plt.xlabel('num epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_epochs) + 1, history['discriminator_acc'], label='discriminator accuracy')
    plt.plot(np.arange(num_epochs) + 1, history['generator_acc'], label='generator accuracy')
    plt.xlabel('num epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train_dcgan(discriminator, generator, criterion, optimizer_discriminator, optimizer_generator, train_data, num_epochs):
    history = defaultdict(list)

    for epoch in range(num_epochs):
        discriminator_loss = 0
        discriminator_acc = 0

        generator_loss = 0
        generator_acc = 0

        start_time = time()
        discriminator.train(True), generator.train(True)
        for real_image in tqdm(train_data):
            real_image = real_image[0].to(device)
            batch_size = real_image.size(0)

            true_predictions = discriminator(real_image).view(-1)
            real_images_loss = criterion(true_predictions, torch.ones(batch_size, device=device))
            real_images_loss.backward()

            noise = torch.randn(batch_size, generator.lvs, 1, 1, device=device)
            fake_image = generator(noise)

            fake_predictions = discriminator(fake_image.detach()).view(-1)
            fake_images_loss = criterion(fake_predictions, torch.zeros(batch_size, device=device))
            fake_images_loss.backward()

            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()

            discriminator_loss += (real_images_loss.cpu().item() + fake_images_loss.cpu().item())

            fake_predictions = (fake_predictions.detach().cpu().numpy() > 0.5) * 1
            true_predictions = (true_predictions.detach().cpu().numpy() > 0.5) * 1
            discriminator_acc += ((fake_predictions == 0).mean() + true_predictions.mean()) / 2

            generator_predictions = discriminator(fake_image).view(-1)
            loss = criterion(generator_predictions, torch.ones(batch_size, device=device))
            loss.backward()

            optimizer_generator.step()
            optimizer_generator.zero_grad()

            generator_loss += loss.cpu().item()
            generator_predictions = (generator_predictions.detach().cpu().numpy() > 0.5) * 1
            generator_acc += generator_predictions.mean()

        discriminator_loss /= len(train_data)
        discriminator_acc /= len(train_data)

        history['discriminator_loss'].append(discriminator_loss)
        history['discriminator_acc'].append(discriminator_acc)

        generator_loss /= len(train_data)
        generator_acc /= len(train_data)

        history['generator_loss'].append(generator_loss)
        history['generator_acc'].append(generator_acc)

        clear_output()
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time() - start_time, 3)))
        print('discriminator loss: {}'.format(np.round(history['discriminator_loss'][-1], 3)))
        print('generator loss: {}'.format(np.round(history['generator_loss'][-1], 3)))

        plot_dcgan_history(history, epoch + 1)

    return discriminator, generator, history


def plot_biggan_history(history, num_epochs):
    plt.figure(figsize=(16, 7))
    plt.plot(np.arange(num_epochs) + 1, history['discriminator_loss'], label='discriminator loss')
    plt.plot(np.arange(num_epochs) + 1, history['generator_loss'], label='generator loss')
    plt.xlabel('num epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train_biggan(discriminator, generator, train_data, num_epochs, n_classes):
    history = defaultdict(list)

    for epoch in range(num_epochs):
        discriminator_loss = 0
        generator_loss = 0

        start_time = time()
        discriminator.train(True), generator.train(True)
        for real_data in tqdm(train_data):
            real_image = real_data[0].to(device)
            batch_size = real_image.size(0)

            real_labels = real_data[1].to(device).long()
            fake_labels = Categorical(torch.tensor([1 / n_classes] * n_classes)).sample([batch_size]).to(device)

            discriminator.optim.zero_grad()

            real_predictions = discriminator(real_image, real_labels)
            noise = torch.randn(batch_size, generator.dim_z, device=device)
            fake_image = generator(noise, generator.shared(fake_labels))

            fake_predictions = discriminator(fake_image.detach(), fake_labels.detach())
            fake_images_loss, real_images_loss = losses.discriminator_loss(fake_predictions, real_predictions)
            loss = (fake_images_loss + real_images_loss)

            loss.backward()
            discriminator.optim.step()
            discriminator_loss += loss.cpu().item()

            generator.optim.zero_grad()

            fake_labels = Categorical(torch.tensor([1 / n_classes] * n_classes)).sample([batch_size]).to(device)
            noise = torch.randn(batch_size, generator.dim_z, device=device)
            fake_image = generator(noise, generator.shared(fake_labels))
            predictions = discriminator(fake_image, fake_labels)
            loss = losses.generator_loss(predictions)

            loss.backward()
            generator.optim.step()
            generator_loss += loss.cpu().item()

        discriminator_loss /= len(train_data)
        generator_loss /= len(train_data)

        history['discriminator_loss'].append(discriminator_loss)
        history['generator_loss'].append(generator_loss)

        clear_output()
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time() - start_time, 3)))
        print('discriminator loss: {}'.format(np.round(history['discriminator_loss'][-1], 3)))
        print('generator loss: {}'.format(np.round(history['generator_loss'][-1], 3)))

        plot_biggan_history(history, epoch + 1)

    return discriminator, generator, history
