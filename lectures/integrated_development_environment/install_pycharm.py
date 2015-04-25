#!/usr/bin/python


import os



# Install Java 7


os.system('sudo apt-get remove openjdk*')

os.system('sudo add-apt-repository ppa:webupd8team/java')

os.system('sudo apt-get update')

os.system('sudo apt-get install oracle-java7-installer')


# Install PyCharm
os.system('http://download.jetbrains.com/python/pycharm-professional-4.0.6.tar.gz')

os.system('sudo mkdir -p /opt/PyCharm')

os.sytem('sudo tar -zxvf pycharm-community-3.0.tar.gz --strip-components 1 -C /opt/PyCharm')