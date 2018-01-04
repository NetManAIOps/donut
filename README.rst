DONUT
=====

Donut is an anomaly detection algorithm for periodic KPIs.

Installation
------------

Checkout this repository and execute:

.. code-block:: bash

    pip install git+https://github.com/thu-ml/zhusuan.git
    pip install git+https://github.com/korepwx/tfsnippet.git
    pip install .

This will first install `ZhuSuan <https://github.com/thu-ml/zhusuan>`_ and
`TFSnippet <https://github.com/korepwx/tfsnippet>`_, the two major dependencies
of Donut, then install the Donut package itself.

API Usage
---------

To prepare the data:

.. code-block:: python

    import numpy as np
    from donut import complete_timestamp, standardize_kpi

    # Read the raw data.
    timestamp, values, labels = ...
    # If there is no label, simply use all zeros.
    labels = np.zeros_like(values, dtype=np.int32)

    # Complete the timestamp, and obtain the missing point indicators.
    timestamp, missing, (values, labels) = \
        complete_timestamp(timestamp, (values, labels))

    # Split the training and testing data.
    test_portion = 0.3
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    # Standardize the training and testing data.
    train_values, mean, std = standardize_kpi(
        train_values, excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

To construct a Donut model:

.. code-block:: python

    import tensorflow as tf
    from donut import Donut
    from tensorflow import keras as K
    from tfsnippet.modules import Sequential

    # We build the entire model within the scope of `model_vs`,
    # it should hold exactly all the variables of `model`, including
    # the variables created by Keras layers.
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=5,
        )

To train the Donut model, and use a trained model for prediction:

.. code-block:: python

    from donut import DonutTrainer, DonutPredictor

    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)

    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        test_score = predictor.get_score(test_values, test_missing)

To save and restore a trained model:

.. code-block:: python

    from tfsnippet.utils import get_variables_as_dict, VariableSaver

    with tf.Session().as_default():
        # Train the model.
        ...

        # Remember to get the model variables after the birth of a
        # `predictor` or a `trainer`.  The :class:`Donut` instances
        # does not build the graph until :meth:`Donut.get_score` or
        # :meth:`Donut.get_training_objective` is called, which is
        # done in the `predictor` or the `trainer`.
        var_dict = get_variables_as_dict(model_vs)

        # save variables to `save_dir`
        saver = VariableSaver(var_dict, save_dir)
        saver.save()

    with tf.Session().as_default():
        # Restore variables from `save_dir`.
        saver = VariableSaver(get_variables_as_dict(model_vs), save_dir)
        saver.restore()


If you need more advanced outputs from the model, you may derive the outputs
by using `model.vae` directly, for example:

.. code-block:: python

    from donut import iterative_masked_reconstruct

    # Obtain the reconstructed `x`, with MCMC missing data imputation.
    # See also:
    #   :meth:`donut.Donut.get_score`
    #   :func:`donut.iterative_masked_reconstruct`
    #   :meth:`tfsnippet.modules.VAE.reconstruct`
    input_x = ...  # 2-D `float32` :class:`tf.Tensor`, input `x` windows
    input_y = ...  # 2-D `int32` :class:`tf.Tensor`, missing point indicators
                   # for the `x` windows
    x = model.vae.reconstruct(
        iterative_masked_reconstruct(
            reconstruct=model.vae.reconstruct,
            x=input_x,
            mask=input_y,
            iter_count=mcmc_iteration,
            back_prop=False
        )
    )
    # `x` is a :class:`tfsnippet.stochastic.StochasticTensor`, from which
    # you may derive many useful outputs, for example:
    x.tensor  # the `x` samples
    x.log_prob(group_ndims=0)  # element-wise log p(x|z) of sampled x
    x.distribution.log_prob(input_x)  # the reconstruction probability
    x.distribution.mean, x.distribution.std  # mean and std of p(x|z)
