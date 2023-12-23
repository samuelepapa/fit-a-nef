Usage
=====

.. _installation:

Install
------------

To use fit-a-nef, you need to install it. To install it, you need to
run the following command in a terminal:

.. code-block:: console

   (.venv) $ pip install .

It is important to note that jax and torch are dependencies to this library, however, they might be installed with the
correct version. If this happens, try installing jax and torch first, and only then run the command above.

Installing the repository
----------------------------

After the installation of the library, you can install the dependencies to run the code in the repository.

The main additional dependencies are :code:`wandb`, :code:`optuna`, and :code:`ml_collections`. To install them, run the following command:

.. code-block:: console

   (.venv) $ pip install wandb optuna ml_collections

The, for the shape dataset we use :code:`trimesh` to render the shapes, and a marching cubes algorithm based on it that
is available in the `dataset/shape_dataset/utils` folder. To install them, run the following command:

.. code-block:: console

   (.venv) $ pip install trimesh
   (.venv) $ cd dataset/shape_dataset/utils
   (.venv) $ python setup.py build_ext --inplace

If problems arise when installing or running the marching cube algorithm, it could be due to the CUDA version being
used. In that case, consider simply removing this plotting from the code. It is possible to visualize shapes by
plotting the point cloud, marching cubes are not necessary.

Now you can look at :ref:fitting_neural_fields for more information on how to actually train your first models.
