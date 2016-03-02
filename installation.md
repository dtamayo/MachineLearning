# Setup before first workshop

Throughout the workshop, we will be using Python and some of its supporting packages.  This is a quick guide to get everything installed ahead of time so we can hit the ground running.

# C compiler and Git

Some Python libraries require a C compiler and we will be using Git to share files and collaborate on projects.  Try typing `gcc` and `git` in a terminal.  If you get 'command not found' for either of them, you need to install them.  If you are on Mac, you can get both by following the instructions here:

http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

If you are on linux, you probably want to install them with either

    sudo yum install gcc git-all

or

    sudo apt-get install gcc git-all

or just google installing git and gcc on your flavor of linux.

# Anaconda

Anaconda is an easy-to-install python distribution that has all the things you will need in one place.  If you are confident with Python, make sure you have the packages listed below installed.  If you are unsure, we recommend installing Anaconda to avoid any conflicts/glitches.  To install, visit 

https://www.continuum.io/downloads 

and follow the instructions for your particular operating system.  You will have the option between python 2.7 and 3.5.  Choose python 2.7 (you always have the option of still using python 3, see below).  

# Make a conda environment

Conda environments let you create an isolated python installation on your machine, which helps keep your installation clean and less likely to run into version clashes between packages.

Open a terminal window and type

    conda create -n ml python=2

This will create a conda environment with name (-n) ml (machine learning--can name whatever you want) that uses the latest version of python 2.  In the same way, you could create a separate conda environment that runs python3 if you ever want to try it.

If you now type

    python

It will launch the default anaconda python.  To use a conda environment, you always have to activate it first (each time you open a terminal window) with

    source activate ml

or whatever the name of your environment is.  You should now see a (ml) in front of the command prompt.  Now if you type `python`, it will launch the python from your conda environment.

Now we have to install the packages we will use for the workshop:

    conda install pandas seaborn scikit-learn jupyter

This will install the packages pandas, seaborn, scikit-learn, and jupyter, along with all their dependencies.  We will talk in the workshop about what each of them do.  If you ever find you need a different python package, you can do

    conda install packagename

# Running Jupyter Notebooks

We will be running everything in the workshop from Jupyter notebooks (this is the new and more general version of ipython notebooks, which you might hear us say--they mean the same thing).  To open a new jupyter notebook, type

    jupyter notebook

This will launch jupyter in your web browswer.  On the right, click the new button, and select Python 2.  In the cell, type

    import pandas
    import seaborn
    import sklearn

Then press shift+enter to execute the cell.  If you do not get an error, then all the installation has worked.  

If you have any errors, try to look around on google for solutions.  We can also try to help you at the workshop, or pair you up with someone that has a working installation.

Quickly read through the short `Overview of the Notebook UI` section at

https://github.com/jupyter/notebook/blob/b3dfa062853a0780a5f818d487910daf11c644fa/docs/source/examples/Notebook/Notebook%20Basics.ipynb

for how to do basic operations with jupyter notebooks.

See you soon!
