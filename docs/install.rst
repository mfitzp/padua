Installation
============

The following instructions should allow you to get PaDuA up and running on your Python installation.

Windows
=======

Install Python 2.7 or 3.4 Windows installer from the `Python download site`_.

You can get Windows binaries for most required Python libraries from `the Pythonlibs library`_.
At a minimum you will need to install NumPy_, SciPy_, `Scikit-Learn`_ and Matplotlib_.
Make sure that the installed binaries match the architecture (32bit/64bit) and the installed Python version.

With those installed you should be able to install the latest release of PaDuA with::

    pip install padua


Windows Using Anaconda
======================

Install Anaconda for Windows. Link to the website is http://continuum.io/downloads.
Make the decision at this point whether to use 64bit or 32bit versions and stick to it.

Anaconda will install many useful packages for you by default. Open the Anaconda command prompt and ensure they are
setup with::

    conda install numpy scipy scikit-learn matplotlib

With those installed you should be able to install the latest release of PaDuA with::

    pip install padua


MacOS X
=======

The simplest approach to setting up a development environment is through the
MacOS X package manager Homebrew_. It should be feasible to build all these tools from
source, but I'd strongly suggest you save yourself the bother.

Install Homebrew as follows::

    ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"

Ensure Python 2.7 or 3.4 is installed::

    brew install python

Or::

    brew install python3

Next use pip to install all required Python packages. This can be done in a one liner with pip::

    pip install numpy scipy pandas matplotlib scikit-learn

With those installed you should be able to install the latest release of PaDuA with::

    pip install padua


MacOS X Using Anaconda
======================

Install Anaconda for MacOS X. Link to the website is http://continuum.io/downloads.

Anaconda will install many useful packages for you by default. Open the Anaconda command prompt and ensure they are
setup with::

    conda install numpy scipy scikit-learn matplotlib

With those installed you should be able to install the latest release of PaDuA with::

    pip install padua


Linux
=====

For Python3 install the following packages::

    sudo apt-get install g++ python3 python3-dev python3-pip git gfortran libzmq-dev
    sudo apt-get install python3-matplotlib python3-requests python3-numpy python3-scipy

You can also install the other packages using pip3 (the names on PyPi are
the same as for the packages minus the python3- prefix). Once installation of the above has completed you're ready to go.

With those installed you should be able to install the latest release of PaDuA with::

    pip3 install padua


.. _Python download site: http://www.python.org/getit/
.. _the Pythonlibs library: http://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _NumPy: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
.. _SciPy: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
.. _Scikit-Learn: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn
.. _Matplotlib: http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
.. _Pip: http://www.lfd.uci.edu/~gohlke/pythonlibs/#pip
.. _Homebrew: http://brew.sh/

