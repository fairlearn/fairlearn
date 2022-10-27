..
    In this file, we provide implementation details and example code for
    adversarial fairness. NOTE: The examples are located at
    "test_othermlpackages/adversarial_fairness.py" for testing, so any changes
    to the examples here should be reflected there.

.. _adversarial:

Adversarial Mitigation
--------------------

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

In :ref:`models`, we discuss the models that this implementation accepts.
In :ref:`data_types`, we discuss the input format of :math:`X,`
how :math:`Y` and :math:`A` are preprocessed, and
how the loss functions :math:`L_P` and :math:`L_A` are chosen.
Finally, in :ref:`training` we give some
useful tips to keep in mind when training this model, as
adversarial methods such as these
can be difficult to train.

.. _models:

Models
~~~~~~

One can implement the predictor and adversarial neural networks as
a `torch.nn.Module` (using PyTorch) or as a `tensorflow.keras.Model` (using TensorFlow).
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

.. _data_types:

Data types and loss functions
~~~~~~~~~~~~~~~~~~

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

.. _training:

Training
~~~~~~~~
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
   the constructor's keyword :code:`callbacks` (see :ref:`Example 2`).
   Optionally, have this function return :code:`True`
   to indicate early stopping.
#. :footcite:t:`zhang2018mitigating` have found it to be useful to maintain a global step
   count and gradually increase :math:`\alpha` while decreasing the learning
   rate :math:`\eta` and taking :math:`\alpha \eta \rightarrow 0`
   as the global step count increases. In particular, use a callback function to perform
   these hyperparameter updates. An example can be seen in the example notebook.


.. _Example 1:

Example 1: Basics & model specification
~~~~~~~~~
First, we cover a most basic application of adversarial mitigation.
We start by loading and preprocessing the dataset::

    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    pos_label = y[0]

    z = X["sex"] # In this example, we consider 'sex' the sensitive feature.

The UCI adult dataset cannot be fed into a neural network (yet),
as we have many columns that are not numerical in nature. To resolve this
issue, we could for instance use one-hot encodings to preprocess categorical
columns. Additionally, let's preprocess the numeric columns to a
standardized range. For these tasks, we can use functionality from
scikit-learn (:py:mod:`sklearn.preprocessor`). We also use an imputer
to get rid of NaN's::

    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from numpy import number

    ct = make_column_transformer(
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("normalizer", StandardScaler()),
                ]
            ),
            make_column_selector(dtype_include=number),
        ),
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
                ]
            ),
            make_column_selector(dtype_include="category"),
        ),
    )

As with other machine learning methods, it is wise to take a train-test split
of the data in order to validate the model on unseen data::

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, y, z, test_size=0.2, random_state=12345, stratify=y
    )

    X_prep_train = ct.fit_transform(X_train) # Only fit on training data!
    X_prep_test = ct.transform(X_test)


Now, we can use :class:`~fairlearn.adversarial.AdversarialFairnessClassifier`
to train on the
UCI Adult dataset. As our predictor and adversary models, we use for
simplicity the default constructors for fully connected neural
networks with sigmoid activations implemented in Fairlearn. We initialize
neural network constructors
by passing a list :math:`h_1, h_2, \dots` that indicate the number of nodes
:math:`h_i` per hidden layer :math:`i`. You can also put strings in this list
to indicate certain activation functions, or just pass an initialized
activation function directly.

The specific fairness
objective that we choose for this example is demographic parity, so we also
set :code:`objective = "demographic_parity"`. We generally follow sklearn API,
but in this case we require some extra kwargs. In particular, we should
specify the number of epochs, batch size, whether to shuffle the rows of data
after every epoch, and optionally after how many seconds to show a progress
update::

    from fairlearn.adversarial import AdversarialFairnessClassifier

    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=[50, "leaky_relu"],
        adversary_model=[3, "leaky_relu"],
        batch_size=2 ** 8,
        progress_updates=0.5,
        random_state=123,
    )

Then, we can fit the data to our model::

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

Finally, we evaluate the predictions. In particular, we trained the
predictor for demographic parity, so we are not only interested in
the accuracy, but also in the selection rate. MetricFrames are a great resource
here::

    predictions = mitigator.predict(X_prep_test)

    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        demographic_parity_difference,
    )
    from sklearn.metrics import accuracy_score

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=Y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test,
    )

Then, to display the result::

    >>> print(mf.by_group)
            accuracy selection_rate
    sex                            
    Female  0.906308       0.978664
    Male    0.723336       0.484927

The above statistics tell us that the accuracy of our model is quite good,
90% for females and 72% for males. However, the selection rates differ, so there
is a large demographic disparity here. When using adversarial fairness
out-of-the-box, users may not yield such
good training results after the first attempt. In general, training
adversarial networks is hard, and users may need to tweak the
hyperparameters continuously. Besides general scikit-learn algorithms
that finetune estimators,
:ref:`Example 2` will demonstrate some problem-specific
techniques we can use such as using dynamic hyperparameters,
validation, and early stopping to improve adversarial training.

.. _Example 2:

Example 2: Finetuning training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial learning is inherently difficult because of various issues,
such as mode collapse, divergence, and diminishing gradients.
In particular, mode collapse seems a real problem on this dataset: the
predictor and adversary trap themselves in a local minimum by favoring one
class (mode). Problems with diverging parameters may also occur, which
may be an indication of a bad choice of hyperparameters, such as a
learning rate that is too large. The problems that a user may encounter are
of course case specific, but general good practices when training
such models are: train slowly, ensuring the
losses remain balanced, and keep track of validation accuracies.
Additionally, we found that single hidden layer neural
networks work best for this use case.

In this example, we demonstrate some of these good practices.
We start by defining our
predictor neural network explicitly so that it is more apparent.
We will be using PyTorch, but the same can be achieved using Tensorflow::

    import torch

    class PredictorModel(torch.nn.Module):
        def __init__(self):
            super(PredictorModel, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(X_prep_train.shape[1], 200),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(200, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.layers(x)


    predictor_model = PredictorModel()

We also take a look at some validation
metrics. Most importantly, we chose the demographic parity difference
to check to what
extent the constraint (demographic parity in this case) is satisfied.
We also look at the selection rate to observe whether our model is
suffering from mode collapse, and we also calculate the accuracy on the
validation set as well.
We will pass this validation step to our model later::

    from numpy import mean

    def validate(mitigator):
        predictions = mitigator.predict(X_prep_test)
        dp_diff = demographic_parity_difference(
            Y_test == pos_label,
            predictions == pos_label,
            sensitive_features=Z_test,
        )
        accuracy = mean(predictions.values == Y_test.values)
        selection_rate = mean(predictions == pos_label)
        print(
            "DP diff: {:.4f}, accuracy: {:.4f}, selection_rate: {:.4f}".format(
                dp_diff, accuracy, selection_rate
            )
        )
        return dp_diff, accuracy, selection_rate

We may define the optimizers however we like. In this case, we use the
suggestion from the paper to set the hyperparameters :math:`\alpha` and learning
rate :math:`\eta` to depend on the timestep such that :math:`\alpha \eta
\rightarrow 0` as the timestep grows::

    schedulers = []

    def optimizer_constructor(model):
        global schedulers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        )
        return optimizer

    step = 1

We make use of a callback function to both update the hyperparameters and to
validate the model. We update these hyperparameters at every 10 steps, and we
validate every 100 steps. Additionally, we can implement early stopping
easily by calling :code:`return True` in a callback function::

    from math import sqrt

    def callbacks(model, *args):
        global step
        global schedulers
        step += 1
        # Update hyperparameters
        model.alpha = 0.3 * sqrt(step // 1)
        for scheduler in schedulers:
            scheduler.step()
        # Validate (and early stopping) every 50 steps
        if step % 50 == 0:
            dp_diff, accuracy, selection_rate = validate(model)
            # Early stopping condition:
            # Good accuracy + low dp_diff + no mode collapse
            if (
                dp_diff < 0.03
                and accuracy > 0.8
                and selection_rate > 0.01
                and selection_rate < 0.99
            ):
                return True

Then, the instance itself. Notice that we do not explicitly define loss
functions, because adversarial fairness is able to infer the loss function
on its own in this example::

    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=[3, "leaky_relu"],
        predictor_optimizer=optimizer_constructor,
        adversary_optimizer=optimizer_constructor,
        epochs=10,
        batch_size=2 ** 7,
        shuffle=True,
        callbacks=callbacks,
        random_state=123,
    )

Then, we fit the model::

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

Finally, we validate as before, and take a look at the results::

    >>> validate(mitigator) # to see DP difference, accuracy, and selection_rate
    (0.12749738693557688, 0.8005937148121609, 0.8286416214556249)
    >>> predictions = mitigator.predict(X_prep_test)
    >>> mf = MetricFrame(
            metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
            y_true=Y_test == pos_label,
            y_pred=predictions == pos_label,
            sensitive_features=Z_test,
        )
    >>> print(mf.by_group)
            accuracy selection_rate
    sex                            
    Female  0.823129       0.743352
    Male    0.789441       0.870849

Notice we achieve a much lower demographic parity
difference than in Exercise 1! This may come at the cost of some accuracy,
but such a tradeoff is to be expected as we are purposely mitigating
the unfairness that was present in the data.

.. _Example 3:

Example 3: Scikit-learn applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AdversarialFairness is quite compliant with scikit-learn API, so functions
such as pipelining and model selection are applicable here. In particular,
applying pipelining might seem complicated as scikit-learn only pipelines
:code:`X` and :code:`Y`, not the :code:`sensitive_features`.
We overcome this issue by passing the sensitive features through the
pipeline as keyword-argument :code:`[name of model]__sensitive_features`
to fit::

    >>> pipeline = Pipeline(
            [
                ("preprocessor", ct),
                (
                    "classifier",
                    AdversarialFairnessClassifier(
                        backend="torch",
                        predictor_model=[50, "leaky_relu"],
                        adversary_model=[3, "leaky_relu"],
                        batch_size=2 ** 8,
                        random_state=123,
                    ),
                ),
            ]
        )
    >>> pipeline.fit(X_train, Y_train, classifier__sensitive_features=Z_train)
    >>> predictions = pipeline.predict(X_test)
    >>> mf = MetricFrame(
            metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
            y_true=Y_test == pos_label,
            y_pred=predictions == pos_label,
            sensitive_features=Z_test,
        )
    >>> print(mf.by_group)
            accuracy selection_rate
    sex                            
    Female  0.906308       0.978664
    Male    0.723336       0.484927

Notice how the same result is obtained as in :ref:`Example 1`.
