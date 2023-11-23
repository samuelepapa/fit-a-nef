from absl.testing import absltest

from tests.initializers import RandomInitTest
from tests.model_inits import TestFourier, TestGabor, TestRFFNet, TestSIREN
from tests.saving_and_loading import (
    TestFlattenAndUnflattenDict,
    TestFlattenAndUnflattenParams,
    TestModelSaveAndLoad,
)
from tests.test_masking import MaskingTest

if __name__ == "__main__":
    absltest.main()
