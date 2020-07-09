r"""Representation (embeddings) generation scripts."""
import tensorflow as tf
import custom_model
import custom_dataset

class RepGen:
    """A representation generator for range of characters.

    """

    def __init__(self, model_name, ckpt_dir):
        """Need a checkpoint directory to initialize RepGen.

        Args:
            ckpt_dir: Str, relative path to TensorFlow checkpoint directory
            model_name: Str, one of the available model names in custom_model.py
        Raises:
            ValueError: if model_name not found
            ValueError: if ckpt_dir don't contain TensorFlow formatted checkpoint
        """
        self.model = None
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir

    @property
    def model_name(self):
        return self.__model_name

    @property
    def ckpt_dir(self):
        return self.__ckpt_dir

    @model_name.setter
    def model_name(self, model_name):
        # Check if model already defined in custom_model
        if model_name not in custom_model.MODLE_MAP.keys():
            self.__model_name = None
            raise ValueError('Model name not found in custom_model module.')

        self.model, _ = custom_model.create_full_model(model_name)
        self.__model_name = model_name
        print(model_name + ' model successfully created.')

    @ckpt_dir.setter
    def ckpt_dir(self, ckpt_dir):
        try:
            ckpt_path = tf.train.latest_checkpoint('ckpts/ResNet50Base')
            self.model.load_weights(ckpt_path)
        except:
            print('Please make sure model and checkpoint are compatible.')
            raise

        self.__ckpt_dir = ckpt_dir
        print('{} model successfully loaded weights from {}.'.format(self.__model_name, self.__ckpt_dir))

    def get_embeddings(self, img_dir):
        """For the image files in 'img_dir', return their embeddings.

        Args:
            img_dir: Str, relative path to directory where all character images are stored

        Returns:
            embeddings: List of (filename, embeddings) pair
        """
        # Get dataset with filename as label
        dataset = custom_dataset.get_filename_dataset(img_dir)

        # Get filenames and their corresponding embeddings
        filenames = []
        embeddings = []
        for img, filename in dataset:
            # decode Tensor into string
            filename_str = filename.numpy()[0].decode('utf-8')
            filenames.append(filename_str)

            # Get embeddings
            embedding = self.model.predict(img)[0]
            embeddings.append(embedding)

        return list(zip(filenames, embeddings))
