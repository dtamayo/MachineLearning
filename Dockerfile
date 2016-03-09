FROM andrewosh/binder-base

# for use with mybinder.org

MAINTAINER Daniel Tamayo <tamayo.daniel@gmail.com>

USER root
COPY . $HOME/

RUN pip install pandas seaborn scikit-learn jupyter pydataset
