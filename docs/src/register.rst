.. module:: overload_numpy.overload

.. _overload_numpy-overload:

#############################################
Override Registry (`overload_numpy.overload`)
#############################################

.. automodule:: overload_numpy.overload



A note about |ufunc|
====================

When registering a |ufunc| override a wrapper object is returned instead of the
original function. On these objects only the ``__call__`` and ``register``
methods are public API.



API
===

.. automodapi:: overload_numpy
    :no-main-docstr:
    :no-heading:
    :noindex:
    :skip: NPArrayOverloadMixin
    :skip: NPArrayFuncOverloadMixin
    :skip: NPArrayUFuncOverloadMixin
