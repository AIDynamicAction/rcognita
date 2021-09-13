About
======================

``rcognita`` Python package is designed for hybrid simulation of agents and environments (generally speaking, not necessarily reinforcement learning agents).
Its main idea is to have an explicit implementation of sampled controls with user-defined sampling time specification.

Installation (basic)
======================

The package can be installed via `pip` by typing in terminal:

``pip install rcognita``

Getting started
======================

The package is organized in modules.

These are:

* ``controllers``

* ``loggers``

* ``models``

* ``simulator``

* ``systems``

* ``utilities``

* ``visuals`` 

There is a collection of main modules (presets) for each agent-environment configuration.

To work with ``rcognita``, use one of the presets by ``python`` running it and specifying parameters.
If you want to create your own environment, fork the repo and implement one in ``systems`` via inheriting the ``System`` superclass.

For developers
======================

In Linux-based OS, to build these wiki docs, run inside cloned repo folder:

```
cd docsrc
make
```

Commit changes.
