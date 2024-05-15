.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to altair-nx's documentation!
=====================================

Introduction
------------

*Draw interactive NetworkX graphs with Altair*

**altair-nx** offers a similar **draw** API to NetworkX but returns Altair Charts instead.

This project started as a fork of `nx-altair <https://github.com/Zsailer/nx_altair>`_ (inactive since 2020)
meant for fixing issues and merging pull requests,
but after a full code-base rework (causing backward and forward incompatibility)
and the implementation of new features (e.g. curved edges, self loops, and greater customisation),
it became reasonable for it to be its own library.

Installation: ``pip install altair-nx``

Every function argument is explained here (and most IDEs will pull from it on hover or for auto-complete),
but the simplest starting point for altair-nx is playing around with the `tutorial notebook <https://github.com/T-Flet/altair-nx/blob/master/examples/altair-nx-tutorial.ipynb>`_).

*PS: if you draw something cool or which could be a good example of using combinations of the various features,
feel free to put it in a notebook and open a pull request with it added to the the* `examples' folder <https://github.com/T-Flet/altair-nx/tree/master/examples>`_.

The functions are grouped as follows:

- :py:mod:`altair_nx.draw_altair`: functions taking a ``NetworkX Graph`` and returning an ``Altair Chart``,
  including of course the main function of the library, i.e. ``draw_networkx``,
  whose very numerous arguments get distributed to the other functions in this module
  (each of which draws a different layer: edges, arrows, nodes, and labels).
- :py:mod:`altair_nx.core`: functions taking a ``NetworkX Graph`` and returning a ``pandas DataFrame``
  appropriate for drawing a particular type of layer;
  each of these functions is called by a corresponding one in ``altair_nx.draw_altair``.




.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents
   :glob:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
..
    * :ref:`search`


