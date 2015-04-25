#!/usr/bin/python


import os



# Install Java 7
os.system('sudo apt-get remove openjdk*')

os.system('sudo add-apt-repository ppa:webupd8team/java')

os.system('sudo apt-get update')

os.system('sudo apt-get install oracle-java7-installer')


# Install PyCharm
os.system('wget http://download.jetbrains.com/python/pycharm-professional-4.0.6.tar.gz')

os.system('sudo mkdir -p /opt/PyCharm')

os.system('sudo tar -zxvf pycharm-professional-4.0.6.tar.gz --strip-components 1 -C /opt/PyCharm')

# Prepare system for execution
try:
	
	os.system('mkdir bin')

except OSError:

	pass

os.sytem('ln -sf /opt/PyCharm/bin/pycharm.sh /home/vagrant/bin/pycharm')
