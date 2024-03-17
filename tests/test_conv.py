from tests.conv_test_base import TestConvBase
from tests.custom_testcase import KerasTestCase as TestCase


class TestGroupConv2D(TestConvBase, TestCase):
    dimension = 2
    transpose = False


class TestGroupConv2DTranspose(TestConvBase, TestCase):
    dimension = 2
    transpose = True


class TestGroupConv3D(TestConvBase, TestCase):
    dimension = 3
    transpose = False


class TestGroupConv3DTranspose(TestConvBase, TestCase):
    dimension = 3
    transpose = True
