..
    In this file, we provide implementation details and example code for
    adversarial fairness. NOTE: The examples are located at
    "test_othermlpackages/adversarial_fairness.py" for testing, so any changes
    to the examples here should be reflected there.

.. _adversarial:

Adversarial Mitigation
======================

.. currentmodule:: fairlearn.adversarial

Fairlearn provides an implementation of the adversarial
mitigation method of :footcite:t:`zhang2018mitigating`.
The input to the method consists of features :math:`X,` labels :math:`Y,`
and sensitive features :math:`A`. The goal is to fit an estimator that
predicts :math:`Y` from :math:`X` while enforcing fairness constraints with
respect to :math:`A`. Both classification and regression
are supported (classes :class:`~fairlearn.adversarial.AdversarialFairnessClassifier` and
:class:`~fairlearn.adversarial.AdversarialFairnessRegressor`) with two types of
fairness constraints: demographic parity and equalized odds.

To train an adversarial mitigation algorithm, the user needs to provide
two neural networks, a predictor network and an adversary network,
with learnable weights :math:`W` and :math:`U,` respectively. The predictor
network is constructed to solve the underlying supervised learning task,
without considering fairness, by minimizing the predictor loss :math:`L_P.`
However, to improve fairness, we do not
only minimize the predictor loss, but we also want to decrease the
adversary's ability to predict the sensitive features from the predictor's
predictions (when implementing demographic parity), or jointly from the predictor's
predictions and true labels (when implementing equalized odds).

Suppose the adversary has the loss term :math:`L_A.` The algorithm
updates adversary weights :math:`U` by descending along the gradient :math:`\nabla_U L_A`.
However, when updating the predictor weights :math:`W`, the algorithm uses

.. math::
    \nabla_W L_P - \text{proj}_{\nabla_W L_A} \nabla_W L_P - \alpha \nabla_W L_A.

instead of just gradient.
Compared with standard stochastic gradient descent, there are two additional terms
that seek to prevent the decrease of the adversary loss. The hyperparameter
:math:`\alpha` specifies the strength of enforcing the fairness constraint.
For details, see :footcite:t:`zhang2018mitigating`.

In :ref:`adversarial_models`, we discuss the models that this implementation accepts.
In :ref:`adversarial_data_types`, we discuss the input format of :math:`X,`
how :math:`Y` and :math:`A` are preprocessed, and
how the loss functions :math:`L_P` and :math:`L_A` are chosen.
Finally, in :ref:`adversarial_training` we give some
useful tips to keep in mind when training this model, as
adversarial methods such as these
can be difficult to train.

.. _adversarial_models:

Models
------

One can implement the predictor and adversarial neural networks as
a `torch.nn.Module` (using PyTorch) or as a `keras.Model` (using TensorFlow).
This implementation has a soft dependency on either PyTorch or TensorFlow, and the user
needs to have installed either one of the two soft dependencies. It is not possible to
mix these dependencies, so a PyTorch predictor with a TensorFlow loss function is not
possible.

It is very important to define the neural network models with no activation function
or discrete prediction function on the final layer. So, for instance, when predicting
a categorical feature that is one-hot-encoded, the neural network should output a
vector of real-valued scores, not the one-hot-encoded discrete prediction::

    predictor_model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    adversary_model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=adversary_model
    )

For simple or exploratory use cases, Fairlearn provides a very basic neural
network builder.
Instead of a neural network model, it is possible to pass a list
:math:`[k_1, k_2, \dots]`, where each :math:`k_i` either indicates
the number of nodes (if :math:`k_i` is an integer) or
an activation function (if :math:`k_i` is a string) or
a layer or activation function instance directly (if :math:`k_i` is
a callable).
However, the number of nodes in the input
and output layer is automatically inferred from data, and the final
activation function (such as softmax for categorical
predictors) is also inferred from data.
So, in the following example, the predictor model is
a neural network with an input layer of
the appropriate number of nodes, a hidden layer with 50 nodes and
ReLU activations, and an output layer with an appropriate activation function.
The appropriate function in case of classification will be softmax for one
hot encoded :math:`Y` and sigmoid for binary :math:`Y`::

    mitigator = AdversarialFairnessClassifier(
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"]
    )

.. _adversarial_data_types:

Data types and loss functions
-----------------------------

We require the provided data :math:`X` to be provided as a matrix
(2d array-like) of floats; this data is directly passed to
neural network models.

Labels :math:`Y` and sensitive features :math:`A` are automatically
preprocessed based on their type: binary data is represented as 0/1,
categorical data is one-hot encoded, float data is left unchanged.

:footcite:t:`zhang2018mitigating` do not explicitly define loss functions.
In :class:`~AdversarialFairnessClassifier` and :class:`~AdversarialFairnessRegressor`,
the loss functions are automatically inferred based on
the data type of the label and sensitive features.
For binary and categorical target variables, the training loss is cross-entropy.
For float targets variables, the training loss is the mean squared error.

To summarize:

.. list-table::
   :header-rows: 1
   :widths: 6 4 6 6 10 6
   :stub-columns: 0

   *  - label :math:`Y`
      - derived label :math:`Y'`
      - network output :math:`Z`
      - probabilistic prediction
      - loss function
      - prediction
   *  - **binary**
      - 0/1
      - :math:`\mathbb{R}`
      - :math:`\mathbb{P}(Y'=1)`
        :math:`\;\;=1/(1+e^{-Z})`
      - :math:`-Y'\log\mathbb{P}(Y'=1)`
        :math:`\;\;-(1-Y')\log\mathbb{P}(Y'=0)`
      - 1 if :math:`Z\ge 0`, else 0
   *  - **categorical**
        (:math:`k` values)
      - one-hot encoding
      - :math:`\mathbb{R}^k`
      - :math:`\mathbb{P}(Y'=\mathbf{e}_j)`
        :math:`\;\;=e^{Z_j}/\sum_{\ell=1}^k e^{Z_{\ell}}`
      - :math:`-\sum_{j=1}^k Y'_j\log\mathbb{P}(Y'=\mathbf{e}_j)`
      - :math:`\text{argmax}_j\,Z_j`
   *  - **continuous**
        (in :math:`\mathbb{R}^k`)
      - unchanged
      - :math:`\mathbb{R}^k`
      - not available
      - :math:`\Vert Z-Y\Vert^2`
      - :math:`Z`

The label is treated as binary if it takes on two distinct :code:`int` or :code:`str` values,
as categorical if it takes on :math:`k` distinct :code:`int` or :code:`str` values (with :math:`k>2`),
and as continuous if it is a float or a vector of floats. Sensitive features are treated similarly.

*Note: currently, all data needs to be passed to the model in the first call
to fit.*

.. _adversarial_training:

Training
--------

Adversarial learning is inherently difficult because of various issues,
such as mode collapse, divergence, and diminishing gradients. Mode collapse
is the scenario where the predictor learns to produce one output, and because
it does this relatively well, it will never learn any other output. Diminishing
gradients are common as well, and could be due to an adversary that is trained
too well in comparison to the predictor.
Such problems
have been studied extensively by others, so we encourage the user to find remedies
elsewhere from more extensive sources. As a general rule of thumb,
training adversarially is best done with a lower and possibly decaying learning
rate while ensuring the
losses remain balanced, and keeping track of validation accuracies every few
iterations may save you a lot of headaches if the model suddenly diverges or
collapses.

Some pieces of advice regarding training with adversarial fairness:

#. For some tabular datasets, we found that single hidden layer neural
   networks are easier to train than deeper networks.
#. Validate your model! Provide this model with a callback function in
   the constructor's keyword :code:`callbacks` (see :ref:`adversarial_Example_2`).
   Optionally, have this function return :code:`True`
   to indicate early stopping.
#. :footcite:t:`zhang2018mitigating` have found it to be useful to maintain a global step
   count and gradually increase :math:`\alpha` while decreasing the learning
   rate :math:`\eta` and taking :math:`\alpha \eta \rightarrow 0`
   as the global step count increases. In particular, use a callback function to perform
   these hyperparameter updates. An example can be seen in the example notebook.

Refer to the following examples for more details:

- :ref:`sphx_glr_auto_examples_plot_adversarial_basics.py`
- :ref:`sphx_glr_auto_examples_plot_adversarial_fine_tuning.py`

.. topic:: References

   .. footbibliography::
