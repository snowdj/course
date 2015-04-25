#!/usr/bin/python
""" Installation script for PyCharm.

    You can execute it by typing in the terminal:

        sudo python install_pycharm.py

    At some point, you will need to press ENTER during the script execution.

    When the script is finished, make sure to update your profile by typing

        source .profile

    in the terminal.

    Afterwards, you can start pycharm by typing

        pycharm

"""

# standard library.
import os
import argparse

# Module-wide variables
FILE_NAME = 'pycharm-professional-4.0.6.tar.gz'

""" Auxiliary functions
"""


def install_pycharm():
    """ Install PyCharm.
    """
    _install_java_runtime_environment()

    _install_pycharm()

    _setup_executable()


def _install_java_runtime_environment():
    """ Install Java Runtime Environment.
    """
    os.system('sudo apt-get remove openjdk*')

    os.system('sudo add-apt-repository ppa:webupd8team/java')

    os.system('sudo apt-get update')

    os.system('sudo apt-get install oracle-java7-installer')


def _install_pycharm():
    """ Install most recent PyCharm Professional.
    """
    os.system('wget http://download.jetbrains.com/python/' + FILE_NAME)

    os.system('sudo mkdir -p /opt/PyCharm')

    os.system('sudo tar -zxvf ' + FILE_NAME +
              ' --strip-components 1 -C /opt/PyCharm')


def _setup_executable():
    """ Prepare system for execution.
    """
    try:

        os.system('mkdir /home/vagrant/bin')

    except OSError:

        pass


os.system('ln -sf /opt/PyCharm/bin/pycharm.sh /home/vagrant/bin/pycharm')

""" Execution of module as script.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                'Installation script for PyCharm on softEcon virtual machine.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.parse_args()

    install_pycharm()