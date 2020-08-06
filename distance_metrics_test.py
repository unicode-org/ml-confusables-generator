# Copyright (C) 2020 and later: Google, Inc.

from distance_metrics import ImgFormat, Distance
import numpy as np
import unittest

class TestVisualGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Temporary image array
        cls.img_rgb_0 = np.zeros((32, 32, 3), dtype=np.uint8)
        cls.img_rgb_1 = np.ones((32, 32, 3), dtype=np.uint8)
        cls.img_gray_0 = np.zeros((32, 32), dtype=np.uint8)
        cls.img_gray_1 = np.ones((32, 32), dtype=np.uint8)
        cls.emb_0 = np.zeros(1000, dtype=np.float64)
        cls.emb_1 = np.ones(1000, dtype=np.float64)

    def test_default_init(self):
        """Test default initialization. When default initialization value
        changes, or any private attribute does not match public attribute, this
        test will fail."""
        dis = Distance()

        self.assertEqual(dis._img_format, ImgFormat.RGB)

    def test_img_format_setter(self):
        # Test setter in initialization
        dis = Distance(img_format=ImgFormat.A8)
        self.assertEqual(dis._img_format, ImgFormat.A8)

        # Test setter after initialization
        dis.img_format = ImgFormat.EMBEDDINGS
        self.assertEqual(dis._img_format, ImgFormat.EMBEDDINGS)

        # Test exception
        with self.assertRaises(TypeError):
            dis.img_format = 5

    def test_rgb_metrics(self):
        # Test RGB metrics
        dis = Distance(ImgFormat.RGB)
        metrics = dis.get_metrics()
        self.assertEqual(metrics['manhattan'](self.img_rgb_0, self.img_rgb_1),
                         255.0)
        self.assertEqual(metrics['manhattan'](self.img_rgb_1, self.img_rgb_1),
                         0)
        self.assertEqual(metrics['sum_squared'](self.img_rgb_0, self.img_rgb_1),
                         1.0)
        self.assertEqual(metrics['sum_squared'](self.img_rgb_1, self.img_rgb_1),
                         0)
        self.assertEqual(metrics['cross_correlation'](self.img_rgb_0,
                                                      self.img_rgb_1),
                         1.0)
        self.assertEqual(metrics['cross_correlation'](self.img_rgb_1,
                                                      self.img_rgb_1),
                         0)

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.img_rgb_0.tolist(), self.img_rgb_1)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.img_gray_0, self.img_rgb_1)
        with self.assertRaises(TypeError):
            metrics['sum_squared'](self.img_rgb_0.tolist(), self.img_rgb_1)
        with self.assertRaises(ValueError):
            metrics['sum_squared'](self.img_gray_0, self.img_rgb_1)
        with self.assertRaises(TypeError):
            metrics['cross_correlation'](self.img_rgb_0.tolist(),
                                         self.img_rgb_1)
        with self.assertRaises(ValueError):
            metrics['cross_correlation'](self.img_gray_0, self.img_rgb_1)


    def test_gray_metrics(self):
        # Test gray metrics
        dis = Distance(ImgFormat.A8)
        metrics = dis.get_metrics()
        self.assertEqual(metrics['manhattan'](self.img_gray_0, self.img_gray_1),
                         255.0)
        self.assertEqual(metrics['manhattan'](self.img_gray_1, self.img_gray_1),
                         0)
        self.assertEqual(metrics['sum_squared'](self.img_gray_0, self.img_gray_1),
                         1.0)
        self.assertEqual(metrics['sum_squared'](self.img_gray_1, self.img_gray_1),
                         0)
        self.assertEqual(metrics['cross_correlation'](self.img_gray_0,
                                                      self.img_gray_1),
                         1.0)
        self.assertEqual(metrics['cross_correlation'](self.img_gray_1,
                                                      self.img_gray_1),
                         0)

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.img_gray_0.tolist(), self.img_gray_1)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.img_rgb_0, self.img_gray_1)
        with self.assertRaises(TypeError):
            metrics['sum_squared'](self.img_gray_0.tolist(), self.img_gray_1)
        with self.assertRaises(ValueError):
            metrics['sum_squared'](self.img_rgb_0, self.img_gray_1)
        with self.assertRaises(TypeError):
            metrics['cross_correlation'](self.img_gray_0.tolist(),
                                         self.img_gray_1)
        with self.assertRaises(ValueError):
            metrics['cross_correlation'](self.img_rgb_0, self.img_gray_1)

    def test_emb_metrics(self):
        # Test embedding metrics
        dis = Distance(ImgFormat.EMBEDDINGS)
        metrics = dis.get_metrics()
        self.assertEqual(metrics['manhattan'](self.emb_0, self.emb_1),
                         1000.0)
        self.assertEqual(metrics['manhattan'](self.emb_1, self.emb_1),
                         0)
        self.assertAlmostEqual(metrics['euclidean'](self.emb_0, self.emb_1),
                         np.sqrt(1000))
        self.assertEqual(metrics['euclidean'](self.emb_1, self.emb_1),
                         0)

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.emb_0.tolist(), self.emb_1)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.emb_0, np.ones(1001))
        with self.assertRaises(TypeError):
            metrics['euclidean'](self.emb_0.tolist(), self.emb_1)
        with self.assertRaises(ValueError):
            metrics['euclidean'](self.emb_0, np.ones(1001))


if __name__ == "__main__":
    unittest.main(verbosity=2)
