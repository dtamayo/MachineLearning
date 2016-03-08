Throughout the workshop, we will be using Python and some of its supporting packages.  This is a quick guide to get everything installed ahead of time so we can hit the ground running.

Unfortunately, none of us have experience with Windows, and all the science projects are likely to require either Mac OS X or linux (indeed most science!).  So if you run Windows on your laptop, you will first have to first install Linux (one option is to boot from a USB stick).  This looks like a good start:

http://www.pcworld.com/article/2955460/operating-systems/dual-booting-linux-with-windows-what-you-need-to-know.html

# Setup before first workshop

# C compiler and Git

You will need a C compiler (but no knowledge of C!) and Git to share workshop files and collaborate on projects.  Try typing `gcc` and `git` in a terminal (all commands below should be typed in a terminal window).  If you get 'command not found' for either of them, you need to install them.  If you are on Mac, you can get both by following the instructions here:

http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

If you are on linux, you probably want to install them with either yum or apt-get (or just google installing git and gcc on your flavor of linux).

Once you have installed git, navigate to the directory where you want to put the workshop folder with all the materials, and do

git clone https://github.com/dtamayo/MachineLearning.git

# Anaconda

Anaconda is an easy-to-install python distribution that has all the things you will need in one place.  If you are confident with Python, make sure you have the packages listed below installed and can import all the libraries in jupyter given in the last section.  Unless you are a python master, (and even if you already have some version of python installed), we strongly recommend installing Anaconda to minimize any conflicts/glitches.  To install, visit 

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
    pip install pydataset

This will install the packages pandas, seaborn, scikit-learn, and jupyter, along with all their dependencies.  We will talk in the workshop about what each of them do.  If you ever find you need a different python package, you can do

    conda install packagename

# Running Jupyter Notebooks

We will be running everything in the workshop from Jupyter notebooks (this is the new and more general version of ipython notebooks, which you might hear us say--they mean the same thing).  To open a new jupyter notebook, first activate the environment

    source activate ml
    
then

    jupyter notebook

This will launch jupyter in your web browswer.  On the right, click the new button, and select Python 2.  In the cell, type

    import pandas
    import seaborn
    import sklearn
    import pydataset

Then press shift+enter to execute the cell.  If you do not get an error, then all the installation has worked.  

If you have any errors, try to look around on google for solutions. 

Quickly read through the short `Overview of the Notebook UI` section at

https://github.com/jupyter/notebook/blob/b3dfa062853a0780a5f818d487910daf11c644fa/docs/source/examples/Notebook/Notebook%20Basics.ipynb

for how to do basic operations with jupyter notebooks.

See you soon!
