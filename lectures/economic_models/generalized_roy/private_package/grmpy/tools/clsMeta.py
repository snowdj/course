""" Meta class for the grmpy package
"""

# standard library
import pickle as pkl
import copy

class MetaCls(object):

    def __init__(self):

        self.is_locked = False

    ''' Meta methods.
    '''
    def get_status(self):
        """ Get status of class instance.
        """

        return self.is_locked

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (self.get_status() is False)

        # Update class attributes.
        self.is_locked = True

        # Finalize.
        self._derived_attributes()

        self._check_integrity()

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging.
        assert (self.get_status() == True)

        # Update class attributes.
        self.is_locked = False

    def get_attr(self, key, deep = False):
        """ Get attributes.
        """
        # Antibugging.
        assert (self.get_status() == True)
        assert (deep in [True, False])

        # Copy requested object.
        if deep:

            attr = copy.deepcopy(self.attr[key])

        else:

            attr = self.attr[key]

        # Finishing.
        return attr

    def set_attr(self, key, value, deep = False):
        """ Get attributes.
        """
        # Antibugging.
        assert (self.get_status() == False)
        assert (key in self.attr.keys())

        # Copy requested object.
        if deep:

            attr = copy.deepcopy(value)

        else:

            attr = value

        # Finishing.
        self.attr[key] = attr

    def _derived_attributes(self):
        """ Calculate derived attributes.
        """

        pass

    def _check_integrity(self):
        """ Check integrity of class instance.
        """

        pass

    def store(self, fileName):
        """ Store class instance.
        """
        # Antibugging.
        assert (self.get_status() == True)
        assert (isinstance(fileName, str))

        # Store.
        pkl.dump(self, open(fileName, 'wb'))
