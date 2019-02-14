import keras.backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam


import numpy as np
np.random.seed(1337)
import math
import matplotlib.pyplot as plt

def genGauss(p,n=1,r=1):
    x = []
    y = []
    for k in range(n):
        x_t, y_t = np.random.multivariate_normal([math.sin(2*k*math.pi/n), math.cos(2*k*math.pi/n)], [[0.0125, 0], [0, 0.0125]], p).T
        x.append(x_t)
        y.append(y_t)

    x=np.array(x).flatten()[:,None]
    y=np.array(y).flatten()[:,None]
    x-=np.mean(x)
    y-=np.mean(y)
    train=np.concatenate((x,y),axis=1)
    return train/(np.max(train)*r)

def clip_weights(model, lower, upper):
    for l in model.layers:
        weights = l.get_weights()
        weights = [np.clip(w, lower, upper) for w in weights]
        l.set_weights(weights)

def exchange_weights(discriminator, critic):
  for disc_l, critic_l in zip(discriminator.layers, critic.layers):
    disc_weights = disc_l.get_weights()
    critic_l.set_weights(disc_weights)


def wasserstein_distance(y_true, y_pred):
    return K.mean(y_true * y_pred)


def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=z_dim, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='linear'))
    return model 


def build_critic():
    model = Sequential()
    model.add(Dense(512, input_dim=2, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=2, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == '__main__':
    
    fig, axarr = plt.subplots(1, 2, figsize=(12,4 ))
    save_dir = 'combined_model_results/'
    epochs = 5000
    switch_epoch = 30
    batch_size = 64
    z_dim = 2
    learning_rate = 5e-5

    critic = build_critic()
    critic.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss=wasserstein_distance
    )
    
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer = Adam(learning_rate),
        loss = 'binary_crossentropy'
    )
    generator = build_generator(z_dim)
    z_vector = Input(shape=(z_dim, ))
    fake_for_critic = generator(z_vector)
    fake_for_discriminator = generator(z_vector)
    critic.trainable = False
    discriminator.trainable = False
    fake_for_critic = critic(fake_for_critic)
    fake_for_discriminator = discriminator(fake_for_discriminator)
    combined_gan = Model(inputs=z_vector, outputs = fake_for_discriminator)
    combined_wgan = Model(inputs=z_vector, outputs=fake_for_critic)
    combined_gan.compile(
        optimizer = Adam(learning_rate),
        loss = 'binary_crossentropy'
    )
    combined_wgan.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss = wasserstein_distance
    )    
    X_train = genGauss(100, 5, 1)
    np.random.shuffle(X_train)
    plot_discriminator_loss =[]
    plot_critic_loss = []
    plot_gen_loss = []
    for epoch in range(epochs):
        print('Epoch :', epoch)
        nb_batches = int(X_train.shape[0] / batch_size)
        epoch_critic_loss = []
        epoch_discriminator_loss = []
        epoch_gen_loss = []
        
        index = 0
        while index < nb_batches:
            if epoch < 5 or epoch % 100 == 0:
                Diters = 100
            else:
                Diters = 5
            
            iter = 0
            critic_loss = []
            discriminator_loss = []
            while index < nb_batches and iter < Diters:
                index += 1
                iter += 1
                Z = np.random.uniform(-1, 1, (batch_size, z_dim))
                generated_images = generator.predict(Z)
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                X = np.concatenate((image_batch, generated_images))
                
                
                if(epoch < switch_epoch):
                  y = np.array([-1] * len(image_batch) + [1] * batch_size)
                  critic_loss.append(-critic.train_on_batch(X, y))
                  clip_weights(critic, -0.01, +0.01)
                else:
                  y = np.array([0] * len(image_batch) + [1] * batch_size)
                  discriminator_loss.append(-discriminator.train_on_batch(X, y))
            if(epoch < switch_epoch):
              epoch_critic_loss.append(sum(critic_loss)/len(critic_loss))
              Z = np.random.uniform(-1, 1, (batch_size, z_dim))
              target = -np.ones(batch_size)
              epoch_gen_loss.append(-combined_wgan.train_on_batch(Z, target))
            else:
              epoch_discriminator_loss.append(sum(discriminator_loss)/len(discriminator_loss))
              Z = np.random.uniform(-1, 1, (batch_size, z_dim))
              target = -np.ones(batch_size)
              epoch_gen_loss.append(-combined_gan.train_on_batch(Z, target))
        
        if(epoch < switch_epoch):
          print('\n WGAN Mode: [Loss_C: {:.6f}, Loss_G: {:.6f}]'.format(np.mean(epoch_critic_loss), np.mean(epoch_gen_loss)))
          plot_critic_loss.append(np.mean(epoch_critic_loss))
          plot_gen_loss.append(np.mean(epoch_gen_loss))
          Z = np.random.uniform(-1, 1, (100, z_dim))
          generated_images = generator.predict(Z, verbose=0)
          fig.suptitle('WGAN Mode - Epoch: {}'.format(epoch))
          axarr[0].set_title('Real Data vs. Generated Data')
          axarr[0].scatter(X_train[:, 0], X_train[:, 1], c = 'red', label = 'Real data', marker = '.')
          axarr[0].scatter(generated_images[:, 0], generated_images[:, 1], c = 'green', label = 'Fake data', marker = '.')
          axarr[0].legend(loc='upper left')
          axarr[1].set_title('Generator & Discriminator & Critic error functions')
          axarr[1].plot(plot_critic_loss, color='red', label = 'Critic loss')
          axarr[1].plot(plot_gen_loss, color='green', label='Generator loss')
          plt.legend(loc='upper left')
          fig.savefig(save_dir + 'frame.jpg')
          fig.savefig(save_dir + 'frame'+str(epoch)+'.jpg')
          axarr[0].clear()
          axarr[1].clear()

        else:
          print('\n Vanilla GAN Mode [Loss_D: {:.6f}, Loss_G: {:.6f}]'.format(np.mean(epoch_discriminator_loss), np.mean(epoch_gen_loss)))
          plot_discriminator_loss.append(np.mean(epoch_discriminator_loss))
          plot_gen_loss.append(np.mean(epoch_gen_loss))
          Z = np.random.uniform(-1, 1, (100, z_dim))
          generated_images = generator.predict(Z)
          fig.suptitle('Vanilla GAN Mode - Epoch: {}'.format(epoch))
          axarr[0].set_title('Real Data vs. Generated Data')
          axarr[0].scatter(X_train[:, 0], X_train[:, 1], c = 'red', label = 'Real data', marker = '.')
          axarr[0].scatter(generated_images[:, 0], generated_images[:, 1], c = 'green', label = 'Fake data', marker = '.')
          axarr[0].legend(loc='upper left')
          plot_merged_losses = np.concatenate( (plot_critic_loss, plot_discriminator_loss), axis = 0)
          axarr[1].set_title('Generator & Discriminator & Critic error functions')
          if(switch_epoch == 0):
            axarr[1].plot(plot_gen_loss, color='green', label = 'Generator loss')
            axarr[1].plot(plot_discriminator_loss, color = 'blue', label = 'Discriminator loss')
          else:
            plot_x = np.array( range(len(plot_gen_loss)) ) 
            plt.plot(plot_x[:switch_epoch], plot_critic_loss, color='red', label = 'Critic loss')
            plt.plot(plot_x[:switch_epoch], plot_gen_loss[:switch_epoch], color='green', label = 'Generator loss')
            plt.plot(plot_x[switch_epoch - 1:], plot_merged_losses[switch_epoch - 1:], color='blue', label = 'Discriminator loss')
            plt.plot(plot_x[switch_epoch - 1:], plot_gen_loss[switch_epoch - 1:], color='black', label = 'Generator loss')
          axarr[1].legend(loc='upper left')
          fig.savefig(save_dir + 'frame.jpg')
          fig.savefig(save_dir + 'frame'+str(epoch)+'.jpg')
          axarr[0].clear()
          axarr[1].clear()