import keras
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import boxcox

from resolv_ml.utilities.bijectors import BatchNormalization, BoxCox

np.random.seed(0)
tf.random.set_seed(0)

tfd = tfp.distributions
tfb = tfp.bijectors

plt.rcParams['figure.figsize'] = (10, 6)

n_samples = 99840

# Data exponential generation
data_dist = tfd.Sample(tfd.Exponential(1.), sample_shape=n_samples)
data = data_dist.sample()
plt.hist(data.numpy(), bins=120, density=True, alpha=0.6, color='blue', label='Samples')
plt.show()

# Data gaussian target
z_dist = tfd.Normal(loc=0., scale=1.)
z_samples = z_dist.sample(n_samples)
plt.hist(z_samples.numpy(), bins=120, density=True, alpha=0.6, color='red', label='Samples')
plt.show()


class NFModel(keras.Model):

    def __init__(self):
        super().__init__()
        self._bijectors = [
            BoxCox(power_init_value=1.,
                   shift_init_value=0.,
                   power_trainable=True,
                   shift_trainable=True),
            BatchNormalization(center=False, scale=False)
        ]

    def build(self, input_shape):
        for bij in self._bijectors:
            bij.build(input_shape + (1,))
        self._bijectors_chain = tfb.Chain(self._bijectors)
        self.nf = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape), scale_diag=tf.ones(input_shape)),
            bijector=self._bijectors_chain
        )

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs = self.nf.bijector.inverse(inputs)
        log_likelihood = self.nf.log_prob(inputs)
        negative_log_likelihood = -tf.reduce_sum(log_likelihood)
        self.add_loss(negative_log_likelihood)
        return outputs


model = NFModel()
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=195,
            decay_rate=0.98,
            staircase=False,
            name="ExponentialDecay"
        )
    ),
    run_eagerly=True
)
result = model.fit(x=data, y=np.zeros(n_samples), epochs=5, batch_size=512)
plt.plot(result.history["loss"])
plt.show()

tr_data, llm_lambda = boxcox(data, lmbda=None)
plt.hist(tr_data, bins=120, density=True, alpha=0.6, color='green', label='Samples')
plt.show()
print(f"Scipy MLE power is: {llm_lambda}")

print(f"Trained Power: {model.nf.bijector.bijectors[0].power.value.numpy()}")
print(f"Trained Shift: {model.nf.bijector.bijectors[0].shift.value.numpy()}")
z_samples = model.nf.sample(n_samples / 512)
plt.hist(tf.reshape(z_samples, [-1]).numpy(), bins=120, density=True, alpha=0.6, color='green', label='Samples')
plt.show()

trans_z_samples = model.nf.bijector.inverse(z_samples)
plt.hist(tf.reshape(trans_z_samples, [-1]).numpy(), bins=120, density=True, alpha=0.6, color='green', label='Samples')
plt.show()
