#!/usr/bin/python
""" This script installs the basic software required for the course Software
    Engineering for Economists.

    You can execute it by typing:

        sudo python basics.py

    If you run into problems or have any further questions, do not hesitate to
    contact us at:

        softecon@policy-lab.org

    As an alternative, you can also type the following lines directly into the
    terminal:

        sudo apt-get install -y libblas-dev liblapack-dev gfortran g++

        sudo apt-get install -y python3-dev python3-pip

        sudo apt-get install -y python3-numpy python3-scipy python3-matplotlib

        sudo apt-get install -y ipython3 ipython3-notebook python3-pandas

    Afterwards make sure to edit the .bashrc file with a text editor and add
    the following line at end of the file.

        alias python=python3

    Again, make sure to call

        source .profile

    into the terminal once you are done.

"""

# standard library
import os

# Set python3 as default
file_ = open('.profile', 'a')
file_.write('\n alias python=python3')
file_.close()

# Install basic system libraries
os.system('sudo apt-get install -y libblas-dev liblapack-dev gfortran g++')

# Install scientific tools for python3
os.system('sudo apt-get install -y python3-dev python3-pip')

science_stack= ['python3-numpy', 'python3-scipy', 'python3-matplotlib',
    'ipython3', 'ipython3-notebook', 'python3-pandas']

for package in science_stack:
    os.system('sudo apt-get install -y ' + package)