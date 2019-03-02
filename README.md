# GAN-WGAN-combined-model

We combined both GAN and WGAN into one model using one generator for both.
We implemented this model using Keras.
![combined_model_frames](https://user-images.githubusercontent.com/26183913/52815020-ddc22e00-309d-11e9-9877-9bb3e142075c.gif)
As we can see in the gif photo this combined model is able to learn how to generate 5 Gaussian distributions only in 200 epochs, unlike WGAN which needs more than 300 epochs to converge ([see WGAN result](https://github.com/dhyaaalayed/wgan-gaussian)) and GAN which is not able to learn all of them because of mode collapse problem ([see GAN result](https://github.com/dhyaaalayed/gan-gaussian))

# Content:
- Combined model architecture
- GANs & WGANs problems
- The goal of the combined model
- Generator, Discriminator and Critic Architectures
- How to train the model

# Combined model architecture:
We use one generator, a discriminator and a critic. all of them has the same number of hidden layers.
<p><img width="926" alt="image" src="https://user-images.githubusercontent.com/26183913/52815208-5d4ffd00-309e-11e9-8a4f-687073837eaf.png"></p>

# GANs & WGANs problems
During the training it can happen that the generator gets into a setting in which it always produces the same things. This is a common error with GANs, commonly referred to as Mode Collapse. Although the Generator learns to foolish the corresponding discriminator, it does not learn to represent the complex real data distribution and remains stuck in a small subspace with extremely little diversity. Research has shown that GANs are susceptible to Mode collapse. (Arjovsky et al., 2017) However, if they manage to approximate the distribution well, this estimate is usually very realistic. Where WGAN loses speed compared to GAN because the Critic has to be trained more often pro iteration, it gains stability.

# The goal of the combined model:
The idea was, that finding a method uses both models might lead to something that compensates some disadvantages of the individual models with help of the advantages the other model. one of the main disadvantages of GANs is the problem of mode collapse. Using WGANs that problem does not occur, however WGANs tend to converge really slow because of training the critic more than the discriminator .Using both models together we hoped to eliminate both disadvantages, while achieving results of at least the same quality as the original models.

# Generator, Discriminator and Critic Architectures:
## Generator Architecture:
It consists of an input layer of 2 neurons for the z vector, 3 hidden layers of 512 neurons and an output layer of 2 neurons activation functions of the 3 hidden layers are Relus and linear for the output layer
<p><img width="453" alt="image" src="https://user-images.githubusercontent.com/26183913/52647282-b4aa6d80-2ee4-11e9-9ac2-7e4aff1ddcce.png"></p>

## Discriminator Architecture:
it consists of an input layer of 2 neurons for the training data, 3 hidden layers of 512 neurons of Relu activation function and an output layer of 1 neuron of sigmoid activation function
<p><img width="460" alt="image" src="https://user-images.githubusercontent.com/26183913/52647390-e9b6c000-2ee4-11e9-804a-a1204f5872c3.png"></p>

## Critic Architecture:
it consists of an input layer of 2 neurons for the training data, 3 hidden layers of 512 neurons of Relu activation function and an output layer of 1 neuron of linear activation function
<p><img width="460" alt="image" src="https://user-images.githubusercontent.com/26183913/52800200-9034c980-307b-11e9-9f16-6461a8266432.png"></p>

# How to train the model:
Write in the console `python combined_model.py`. The results will be stored in `combined_model_results/` folder

# License

MIT License

Copyright (c) 2019 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.