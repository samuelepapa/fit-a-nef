Hyperparameter Tuning
=====================

With `fit-a-nef` you can easily set up a hyperparameter tuning experiment.

In the repository, you can see tuning examples for both image and shape datasets.

We use `optuna` to perform the tuning. You can find more information about `optuna` [here](https://optuna.readthedocs.io/en/stable/).

Structure
---------

We define different `study_objective` files (inside the `study_objectives` folder), where we define the objective that optuna will optimize.
Then, a `tune.py` is used to select the correct `study_objective` and run the tuning with the correct sampler and the optimization direction.

This separation between study objective and tuning script allows for full flexibility on the exact parameters that are tuned and the way they are tuned.
This is done at the cost of having duplicate code in the `study_objective` files. When possible, just create a shared utility file with the shared methods
that several study objectives share.

Running
-------

To run the tuning, just run the `tune.py` script with the correct arguments. For example:

.. code-block:: console

    PYTHONPATH=. python tasks/image/tune.py --task=config/image.py:tune --task.study_objective=simple_image --task.optuna.num_trials=100
