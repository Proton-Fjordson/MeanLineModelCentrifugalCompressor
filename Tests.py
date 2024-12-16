import unittest
import compressor
import numpy as np

class CompressorUnitTests(unittest.TestCase):
    def setUp(self):
        self.compressor = compressor.Compressor([], 1.5, 2*np.pi)

    def teardown(self):
        self.compressor.dispose()
    
    def test_return_speed_Hz(self):
        self.assertAlmostEqual(self.compressor.return_speed_Hz(), 1, 10)

    def test_return_speed_rpm(self):
        print(self.compressor._speed)
        self.assertAlmostEqual(self.compressor.return_speed_rpm(), 60, 10)

    def test_massflow(self):
        self.assertEqual(self.compressor._massflow, 1.5)

if __name__ == "__main__":
    unittest.main(verbosity= 2)