Version 0.1
===========

New Features
------------

- ``NDFunctionMixin`` now passes the calling type, not the dispach type to
  ``_Assists`` wrappers. [#22]

- ``NumPyFunctionOverloader``'s keys are now the function's module + name, since
  NumPy sometimes swaps the function behind the scenes, making it a poor choice
  of key. [#22]

- ``_NumPyInfo`` is renamed to ``_NumPyFuncOverloadInfo`` [#16]

- ``dispatch_on`` type information is added to the ``_NumPyFuncOverloadInfo`` object to
  which it dispaches. [#21]

- Arguments to ``TypeConstraint.validate_types`` (and subclasses) are now
  positional-only. [#21]


0.0.1
=====

TODO
