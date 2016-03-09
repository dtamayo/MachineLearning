FROM andrewosh/binder-base

# for use with mybinder.org

MAINTAINER Daniel Tamayo <tamayo.daniel@gmail.com>

USER root
COPY . $HOME/

RUN $HOME/anaconda2/envs/python2/bin/pip install pandas seaborn scikit-learn jupyter pydataset
