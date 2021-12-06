
.. _adversarial:

Adversarial Learning
--------------------

.. currentmodule:: fairlearn.adversarial

Fairlearn provides an implementation of the adversarial
learning method presented in [#4]_.
We assume we have data :math:`X, A, Y`, where we want to
predict :math:`Y` from :math:`X` while being fair with respect to :math:`A`
based on some fairness measure. We firstly create predictor and adversary neural
networks with learnable weights :math:`W` and :math:`U` respectively. Without
considering fairness yet, this typical supervised-learning problem aims to
minimize the predictor loss :math:`L_P`. Now, to improve fairness, we not
only want to minimize the predictor loss, but we want to decrease the
adversary's ability to predict the sensitive features from the predictor's
predictions. Suppose the adversary has loss term :math:`L_A`, then the paper
trains the predictor with gradient:

.. math::
    \nabla_W L_P - \text{proj}_{\nabla_W L_A} \nabla_W L_P - \alpha \nabla_W L_A

If this model converges properly,
the adversary will attain a loss equal to the entropy, so the adversary
can not
predict the sensitive features from the predictions.
Moreover, this model can be trained for either *demographic parity* or
*equalized odds*. If only the predictions are fed to the adversary, the
predictor will learn to satisfy demographic parity. If also the true targets are
fed to the adversary, the predictor will learn to satisfy equalized odds.
For details, see [#4]_.

Firstly, we will dicuss the models that this implementation accepts in
:ref:`models`.
Secondly, in :ref:`data_preprocessing`, we discuss the required
data preprocessing, as :math:`X` must be an array of
floats. Then, in :ref:`loss_functions`,
we describe what are good choices of :math:`L_P` and :math:`L_A`.
Lastly, in :ref:`training` we give some
useful tips to keep in mind when training this model, as
adversarial methods such as these
are inherently difficult to train.

.. _models:

Models
~~~~~~

One can implement the predictor and adversarial neural networks as
a `torch.nn.Module` (using PyTorch) or as a `tensorflow.keras.Model` (using TensorFlow).
This implementation has a soft dependency on either PyTorch or TensorFlow, and the user
needs to have installed either one of the two soft dependencies. It is not possible to
mix these dependencies, so a PyTorch predictor with TensorFlow loss functions are not
possible.

It is very important to define the neural network models with no activation function
or discrete prediction function on the final layer. So, for instance, when predicting
a categorical feature that is one-hot-encoded, the neural network should output the
logits, not the one-hot-encoded discrete prediction.

.. testcode::

    predictor_model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    adversary_model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    mitigator = AdversarialFairness(
        predictor_model=predictor_model,
        adversary_model=adversary_model
    )

For simple or exploratory use cases, Fairlearn provides a very basic neural
network builder.
Instead of a neural network model, we can pass a list of keywords
:math:`k_1, k_2, \dots` that indicate either
the number of nodes :math:`k_i` (if :math:`k_i` is integer) or a keyword
that indicates an activation function (if :math:`k_i` is a string) or
a layer or activation function instance directly (if :math:`k_i` is callable).
However, the number of nodes in the input
and output layer are automatically inferred from data, and the final activation
function (such as softmax for categorical predictors) are inferred from data.
So, in the following example, the predictor model is
a neural network with an input layer of
the appropriate number of nodes, a hidden layer with 50 nodes and
ReLU activations, and an output layer with an appropriate activation function.

.. testcode::

    mitigator = AdversarialFairness(
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"]
    )

Though, we
do recommend you create your own neural networks beforehand,
as you can tweak these to your
liking.

.. _data_preprocessing:

Data Preprocessing
~~~~~~~~~~~~~~~~~~

In contrast to other mitigation techniques, we require you to
preprocess your data :math:`X`
to matrices (2d array-like) of floats. It is very typical to be dealing
with data other than numbers,
such as categorical data, and in those cases we leave it to the
user to preprocess this
to numbers. The user can make the decision on how to encode this column,
for instance
using a one-hot-encoding or by mapping the values to an integer range.
Another reason
that we do not provide such a data preprocessor out of the box,
is that scikit-learn makes
preprocessing easy. For instance, say we
want to encode the categorical data using one-hot encodings,
and we want to scale the other features to a standard range.
An easy way to do this, using sci-kit learn, is
to create a :py:class:`ColumnTransform` with column selectors that apply
:py:class:`OneHotEncoding` and :py:class:`StandardScalar` transforms.

.. doctest::

    >>> from sklearn.compose import make_column_transformer, make_column_selector
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>>
    >>> ct = make_column_transformer(
    ...     (StandardScaler(),
    ...      make_column_selector(dtype_include='number')),
    ...     (OneHotEncoder(drop='if_binary', sparse=False),
    ...      make_column_selector(dtype_include="category")))
    >>>
    >>> X_transformed = ct.fit_transform(X)

Similarly to other methods, there are more common
aspects to preprocessing besides
encoding everything with numbers. In particular, one needs to
take care in handling NaN's,
as NaN's are not supported by PyTorch or TensorFlow. Also,
as for other neural network
models, it helps to scale every feature of the data to a similar range.


.. _loss_functions:

Loss functions
~~~~~~~~~~~~~~

In [#4]_, loss functions are not defined explicitely.
To accomodate and provide the most general interface, you can pass your
own loss functions. This can be either PyTorch or TensorFlow loss functions,
depending on which module the neural networks are. For instance, in the
following example we set the :code:`predictor_loss` to a PyTorch-implemented
binary cross-entropy, and the :code:`adversary_loss` to the mean squared error.
Additionally, we explicitely specify how the **discrete predictions** are
computed, by providing a :code:`predictor_function`. In this case,
the predictor function maps sigmoid logits to the discrete prediction.

.. testcode::

    mitigator = AdversarialFairness(
        backend="torch",
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"]
        predictor_loss=torch.nn.BCEWithLogitsLoss(),
        adversary_loss=torch.nn.MSELoss(),
        predictor_function=lambda pred: (pred >= 0.).astype(float)
    )

*Note that the PyTorch and TensorFlow modules switch the order of arguments in
their loss functions, so be extra careful when defining custom loss functions
so that you adhere to the module-specific argument order*

We do, however, attempt to infer an appropriate loss function for the data.
For instance, if the data appears categorical, categorical cross
entropy loss is used, along with a softmax on the output layer of the model,
and an argmax for the discrete prediction function. 

We also offer some shortcuts for binary, categorical, and continuous data
that you may use to further specify appropriate types. This is useful when
you want to be sure that the inferred distribution type is what you expect.
For instance, to indicate that the predictions should be categorical, you
pass :code:`predictor_loss='category'`
and :code:`prediction_function='category'`.
For a list of keywords and their
implications, see the Table below. 
.. Additionally, there are keywords
.. :code:`auto` and :code:`classification`. The former indicates to automatically
.. infer the distribution (which is the default keyword), and the latter indicates
.. to select either :code:`binary` or :code:`category`.

.. list-table::
   :header-rows: 1
   :widths: 5 5 12 8 11
   :stub-columns: 1

   *  - keyword
      - distribution assumption
      - requirements
      - loss
      - predictor function
   *  - :code:`'continuous'`
      - Normal
      - There are non-integer numbers.
      - Mean squared error
      - identity function
   *  - :code:`'binary'`
      - Binomial
      - There are only two unique values that are integers or strings.
      - Binary cross-entropy
      - 1 if :math:`\hat y \geq \frac12`, else 0
   *  - :code:`'category'`
      - Mulitnomial
      - There are integers or strings.
      - Categorical cross-entropy
      - argmax :math:`\hat y`

*Note: currently, all data needs to be passed to the model in the first call
to fit. If you really need to circumvent this, you can manually use*
:code:`AdversarialFairness.__setup(X, Y, A)` *with a portion of the full
dataset that contains at least every class once.*

Theoretical results of [#4]_ show that under some strong assumptions,
including which loss we take,
the predictor will satisfy the fairness constraint: demographic
parity or equality of odds.

In particular, to state the results of *Proposition 2* of [#4]_
regarding demographic parity, if we assume: 

#. The variables :math:`Y` and :math:`A` have underlying distributions 
   :math:`D_Y` and :math:`D_A` respectively.
#. The adversary is strong enough that, at convergence, it has
   learned a function that minimizes the cross-entropy loss.
#. The predictor completely fools the adversary. In other words, the loss
   of the adversarial is maximized and equal to the entropy of :math:`A`.

Then, the predictor satisfies demographic parity. *This tells us that
choosing cross-entropy loss for the adversary makes sense!*

The specific cross-entropy loss is then still dependent on the assumption we
make on the distribution :math:`D_A`. If we assumed :math:`D_A` follows
a Bernoulli distribution, then binary cross-entropy may be sensible. If
we assume :math:`D_A` is normally distributed, then cross-entropy loss between
two normal distributions is most sensible. Practically, mean squared error loss
is a sensible choice too, as it is proportional to the cross-entropy of two
normal distributions.

.. _training:

Training
~~~~~~~~
Adversarial Learning is inherently difficult because of various issues,
such as mode collapse, divergence, and diminishing gradients. 
Such problems
have been studied extensively by others, so we encourage you to find remedies
elsewhere from more extensive sources. As a general rule of thumb,
training adversarially is best done slowly while ensuring the
losses remain balanced, and keeping track of validation accuracies every few
iterations may save you a lot of headaches.

Additionally, we can provide two specific pieces of advice regarding training
this specific model.

#. For some tabular datasets, we found that single hidden layer neural
   networks are easier to train than deeper networks.
#. Validate your model! Provide this model with a callback function in
   the constructor's keyword :code:`callback_fn`. Optionally, have
   this function return :code:`True` to indicate early stoppings.
#. The authors of [#4]_ have found it to be useful to maintain a global step
   count and gradually increase :math:`\alpha` while decreasing the learning
   rate :math:`\mu` and maintaing :math:`\alpha \cdot \mu \rightarrow 0`
   as the timestep grows. In particular, use a callback function to perform
   these hyperparameter updates. An example can be seen in the example notebook.