# Copyright (C) 2020 and later: Google, Inc.

from distance_metrics import ImgFormat, Distance
import numpy as np
import unittest

class TestVisualGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Temporary image array

        # img_rgb_0 is a 32 x 32 x 3 image with all pixels set to 0
        cls.img_rgb_0 = np.zeros((3, 3, 3), dtype=np.uint8)
        # img_rgb_255 is a 32 x 32 x 3 image with all pixels set to 1
        cls.img_rgb_255 = np.ones((3, 3, 3), dtype=np.uint8) * 255
        # img_rgb_topleft is a 32 x 32 x 3 image with a square on the top-left
        # corner (1 x 1 x 3) set to 100
        cls.img_rgb_topleft = cls.img_rgb_255.copy()
        cls.img_rgb_topleft[:1, :1] = 100
        # img_rgb_botright is a 32 x 32 x 3 image with a square on the
        # bottom-right corner (1 x 1 x 3) set to 200
        cls.img_rgb_botright = cls.img_rgb_255.copy()
        cls.img_rgb_botright[2:, 2:] = 200

        # img_gray_0 is a 32 x 32 image with all pixels set to 0
        cls.img_gray_0 = np.zeros((3, 3), dtype=np.uint8)
        # img_gray_255 is a 32 x 32 image with all pixels set to 1
        cls.img_gray_255 = np.ones((3, 3), dtype=np.uint8) * 255
        # img_gray_topleft is a 32 x 32 image with a square on the top-left
        # corner (2 x 2) set to 1
        cls.img_gray_topleft = cls.img_gray_255.copy()
        cls.img_gray_topleft[:1, :1] = 100
        # img_gray_botright is a 32 x 32 image with a square on the top-left
        # corner (2 x 2) set to 1
        cls.img_gray_botright = cls.img_gray_255.copy()
        cls.img_gray_botright[2:, 2:] = 200

        cls.emb_0 = np.zeros(3, dtype=np.float64)
        cls.emb_1 = np.ones(3, dtype=np.float64)
        cls.emb_123 = np.array([1, 2, 3], dtype=np.float64)

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

        # Test manhattan distance
        self.assertEqual(metrics['manhattan'](self.img_rgb_0, self.img_rgb_255),
                         255.0)
        self.assertEqual(metrics['manhattan'](self.img_rgb_255, self.img_rgb_0),
                         255.0)
        self.assertAlmostEqual(metrics['manhattan'](self.img_rgb_topleft,
                                                    self.img_rgb_botright),
                               (55 + 155) / (3 * 3))
        self.assertEqual(metrics['manhattan'](self.img_rgb_255,
                                              self.img_rgb_255), 0)

        # Test sum squared distance
        self.assertEqual(metrics['sum_squared'](self.img_rgb_0,
                                                self.img_rgb_255), 1.0)
        self.assertEqual(metrics['sum_squared'](self.img_rgb_0,
                                                self.img_rgb_topleft),
                         1.0)
        self.assertAlmostEqual(metrics['sum_squared'](self.img_rgb_topleft,
                                                      self.img_rgb_botright), 
                               0.049633607)
        self.assertAlmostEqual(metrics['sum_squared'](self.img_rgb_255,
                                                      self.img_rgb_botright),
                               0.005283143)
        self.assertEqual(metrics['sum_squared'](self.img_rgb_255,
                                                self.img_rgb_255), 0)

        # Test cross correlation distance
        self.assertEqual(metrics['cross_correlation'](self.img_rgb_0,
                                                      self.img_rgb_255),
                         0)
        self.assertAlmostEqual(metrics['cross_correlation'](
            self.img_rgb_topleft, self.img_rgb_botright), 0.9755619)
        self.assertAlmostEqual(metrics['cross_correlation'](
            self.img_rgb_255, self.img_rgb_botright), 0.99759716)
        self.assertEqual(metrics['cross_correlation'](self.img_rgb_255,
                                                      self.img_rgb_255), 1.0)

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.img_rgb_0.tolist(), self.img_rgb_255)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.img_gray_0, self.img_rgb_255)
        with self.assertRaises(TypeError):
            metrics['sum_squared'](self.img_rgb_0.tolist(), self.img_rgb_255)
        with self.assertRaises(ValueError):
            metrics['sum_squared'](self.img_gray_0, self.img_rgb_255)
        with self.assertRaises(TypeError):
            metrics['cross_correlation'](self.img_rgb_0.tolist(),
                                         self.img_rgb_255)
        with self.assertRaises(ValueError):
            metrics['cross_correlation'](self.img_gray_0, self.img_rgb_255)


    def test_gray_metrics(self):
        # Test gray metrics
        dis = Distance(ImgFormat.A8)
        metrics = dis.get_metrics()

        # Test manhattan distance
        self.assertEqual(
            metrics['manhattan'](self.img_gray_0, self.img_gray_255), 255.0)
        self.assertEqual(
            metrics['manhattan'](self.img_gray_255, self.img_gray_0), 255.0)
        self.assertEqual(
            metrics['manhattan'](self.img_gray_topleft, self.img_gray_botright),
            (55 + 155) / (3 * 3))
        self.assertEqual(
            metrics['manhattan'](self.img_gray_255, self.img_gray_255), 0)

        # Test sum squared distance
        self.assertEqual(metrics['sum_squared'](self.img_gray_0,
                                                self.img_gray_255), 1.0)
        self.assertAlmostEqual(metrics['sum_squared'](self.img_gray_topleft,
                                                      self.img_gray_botright),
                               0.049633607)
        self.assertAlmostEqual(metrics['sum_squared'](self.img_gray_255,
                                                      self.img_gray_botright),
                               0.005283143)
        self.assertEqual(metrics['sum_squared'](self.img_gray_255,
                                                self.img_gray_255), 0)

        # Test cross correlation distance
        self.assertEqual(metrics['cross_correlation'](self.img_gray_0,
                                                      self.img_gray_255), 0)
        self.assertAlmostEqual(metrics['cross_correlation'](
            self.img_gray_topleft, self.img_gray_botright), 0.9755619)
        self.assertAlmostEqual(metrics['cross_correlation'](
            self.img_gray_255, self.img_gray_botright), 0.99759716)
        self.assertEqual(metrics['cross_correlation'](self.img_gray_255,
                                                      self.img_gray_255), 1.0)

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.img_gray_0.tolist(), self.img_gray_255)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.img_rgb_0, self.img_gray_255)
        with self.assertRaises(TypeError):
            metrics['sum_squared'](self.img_gray_0.tolist(), self.img_gray_255)
        with self.assertRaises(ValueError):
            metrics['sum_squared'](self.img_rgb_0, self.img_gray_255)
        with self.assertRaises(TypeError):
            metrics['cross_correlation'](self.img_gray_0.tolist(),
                                         self.img_gray_255)
        with self.assertRaises(ValueError):
            metrics['cross_correlation'](self.img_rgb_0, self.img_gray_255)

    def test_emb_metrics(self):
        # Test embedding metrics
        dis = Distance(ImgFormat.EMBEDDINGS)
        metrics = dis.get_metrics()

        # Test manhattan distance
        self.assertEqual(metrics['manhattan'](self.emb_0, self.emb_1),
                         3)
        self.assertEqual(metrics['manhattan'](self.emb_1, self.emb_1),
                         0)
        self.assertAlmostEqual(metrics['manhattan'](self.emb_123, self.emb_1),
                               3)
        self.assertAlmostEqual(metrics['manhattan'](self.emb_0, self.emb_123),
                               6)

        # Test euclidean distance
        self.assertAlmostEqual(metrics['euclidean'](self.emb_0, self.emb_1),
                         np.sqrt(3))
        self.assertEqual(metrics['euclidean'](self.emb_1, self.emb_1),
                         0)
        self.assertAlmostEqual(metrics['euclidean'](self.emb_123, self.emb_0),
                               np.sqrt((1 ** 2) + (2 ** 2) + (3 ** 2)))

        # Test exception
        with self.assertRaises(TypeError):
            metrics['manhattan'](self.emb_0.tolist(), self.emb_1)
        with self.assertRaises(ValueError):
            metrics['manhattan'](self.emb_0, np.ones(4))
        with self.assertRaises(TypeError):
            metrics['euclidean'](self.emb_0.tolist(), self.emb_1)
        with self.assertRaises(ValueError):
            metrics['euclidean'](self.emb_0, np.ones(4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
