from unittest import TestCase

from keras import ops


class KerasTestCase(TestCase):
    def assertAllEqual(self, a, b, msg=None):
        self.assertTrue(ops.all(ops.equal(a, b)), msg)

    def assertAllLess(self, a, b, msg=None):
        self.assertTrue(ops.all(ops.less(a, b), msg))
