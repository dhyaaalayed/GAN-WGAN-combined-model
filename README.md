# GAN-WGAN-combined-model

We combined both GAN and WGAN into one model using one generator for both.
We implemented this model using Keras.
![combined_model_frames](https://user-images.githubusercontent.com/26183913/52815020-ddc22e00-309d-11e9-9877-9bb3e142075c.gif)

# Content:
- Combined model architecture
- GANs & WGANs problems
- The goal of the combined model
- How to train the model

# Combined model architecture:
We use one generator, a discriminator and a critic. all of them has the same number of hidden layers.
<p><img width="926" alt="image" src="https://user-images.githubusercontent.com/26183913/52815208-5d4ffd00-309e-11e9-8a4f-687073837eaf.png"></p>

# GANs & WGANs problems
During the training it can happen that the generator gets into a setting in which it always produces the same things. This is a common error with GANs, commonly referred to as Mode Collapse. Although the Generator learns to foolish the corresponding discriminator, it does not learn to represent the complex real data distribution and remains stuck in a small subspace with extremely little diversity. Research has shown that GANs are susceptible to Mode collapse. (Arjovsky et al., 2017) However, if they manage to approximate the distribution well, this estimate is usually very realistic. Where WGAN loses speed compared to GAN because the Critic has to be trained more often pro iteration, it gains stability.

# The goal of the combined model:
The idea was, that finding a method uses both models might lead to something that compensates some disadvantages of the individual models with help of the advantages the other model. one of the main disadvantages of GANs is the problem of mode collapse. Using WGANs that problem does not occur, however WGANs tend to converge really slow because of training the critic more than the discriminator .Using both models together we hoped to eliminate both disadvantages, while achieving results of at least the same quality as the original models.

# How to train the model:
Write in the console `python combined_model.py`. The results will be stored in `combined_model_results/` folder