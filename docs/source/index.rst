Welcome to fit-a-nef's documentation!
=====================================

:code:`fit-a-nef` (/fit a nɛf/) is a Python library for quick fitting of thousands of neural fields to entire datasets.

Using the ability of JAX to easily parallelize the operations on a GPU with :code:`vmap`, a sizeable set of neural
fields can be fit to distinct samples at the same time.

The :code:`fit-a-nef` library is designed to easily allow the user to add their own *training task*, *dataset*, and *model*.
It provides a uniform format to store and load large amounts of neural fields in a platform-agnostic way.
Whether you use PyTorch, JAX or any other framework, the neural fields can be loaded and used in your project.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the library and the dependencies for the repository.

:code:`fit-a-nef` is developed by the :ref:`team` at the University of Amsterdam.

.. note::

   Please help us by contributing to the project! See the GitHub repository for more information.

Contents
--------

.. toctree::

   usage
   basic_fitting
   hyperparameter_tuning
   api
   team
