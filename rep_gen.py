# Copyright (C) 2020 and later: Google, Inc.

import tensorflow as tf
import numpy as np
import custom_model
import custom_dataset
import pickle
import os

class RepGen:
    """A representation generator for range of characters.

    """

    def __init__(self, model_name, ckpt_dir, out_dir="embeddings",):
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
        self.out_dir = out_dir

    @property
    def model_name(self):
        return self.__model_name

    @property
    def ckpt_dir(self):
        return self.__ckpt_dir

    @property
    def out_dir(self):
        return self.__out_dir

    @model_name.setter
    def model_name(self, model_name):
        # Check if model already defined in custom_model
        if model_name not in custom_model.MODEL_MAP.keys():
            self.__model_name = None
            raise ValueError('Model name not found in custom_model module.')

        self.model, _ = custom_model.create_full_model(model_name)
        self.__model_name = model_name
        print(model_name + ' model successfully created.')

    @ckpt_dir.setter
    def ckpt_dir(self, ckpt_dir):
        try:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            self.model.load_weights(ckpt_path)
        except:
            print('Please make sure model and checkpoint are compatible.')
            raise

        self.__ckpt_dir = ckpt_dir
        print('{} model successfully loaded weights from {}.'.format(self.__model_name, self.__ckpt_dir))

    @out_dir.setter
    def out_dir(self, out_dir):
        self.__out_dir = out_dir

    def get_embeddings(self, img_dir):
        """For the image files in 'img_dir', return their embeddings.

        Args:
            img_dir: Str, relative path to directory where all character images are stored

        Returns:
            embeddings: List of (Unicode, embeddings) pair
        """
        # Get dataset with filename as label
        dataset = custom_dataset.get_filename_dataset(img_dir)

        # Get unicode code points and their corresponding embeddings
        codepoints = []
        embeddings = []
        i = 0
        print('Generating embeddings...')
        for img, filename in dataset:
            i += 1
            if i % 100 == 0:
                print("Getting embedding #" + str(i) + ".")
            # decode Tensor into string
            filename_str = filename.numpy()[0].decode('utf-8')
            codepoints.append(filename_str.split("_")[0])

            # Get embeddings
            embedding = self.model.predict(img)[0]
            embeddings.append(embedding)

        return codepoints, embeddings

    def write_embeddings_from_image(self, img_dir, out_file, char_as_label=True):
        """Get embeddings and write embeddings and labels to .tsv files. This function will write to two .tsv files:
        [out_file]_vec.tsv and [out_file]_meta.tsv.

        Args:
            img_dir: Str, relative path to directory where all character images are stored
            out_file: Str, name of the output file intended to write to
            char_as_label: Bool, whether
        """
        # Get absolute directory path
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        self.__check_create_out_dir_abs(out_dir_abs)
        out_file_abs = os.path.join(out_dir_abs, out_file)
        out_file_vec_abs = out_file_abs + '_vec.tsv'
        out_file_meta_abs = out_file_abs + '_meta.tsv'

        # Get filename dataset
        ds = custom_dataset.get_filename_dataset(img_dir)

        # Get model predictions and unicode codep oints
        codepoints, embeddings = self.get_embeddings(img_dir=img_dir)


        # Write embeddings to file
        print("Writing embeddings to file {}...".format(out_file_vec_abs))
        np.savetxt(out_file_vec_abs, embeddings, delimiter='\t')
        print('Successfully written to file {}.'.format(out_file_vec_abs))

        # Change Unicode code point to character if specified
        if char_as_label:
            codepoints = [chr(int('0x'+codepoint[2:], 16)) for codepoint in codepoints]

        # Write labels
        print("Writing embeddings to file {}...".format(out_file_meta_abs))
        with open(out_file_meta_abs, "w+") as f_out:
            for label in codepoints:
                f_out.write(label)
                f_out.write('\n')
        print('Successfully written to file {}.'.format(out_file_meta_abs))

    def write_embeddings_from_list(self, codepoints, embeddings, out_file, char_as_label=True):
        """Write labels and embeddings to file.

        Args:
            codepoints: Str, relative path to directory where all character images are stored
            out_file: Str, name of the output file intended to write to
            char_as_label: Bool, whether
        """
        # Get absolute directory path
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        self.__check_create_out_dir_abs(out_dir_abs)
        out_file_abs = os.path.join(out_dir_abs, out_file)
        out_file_vec_abs = out_file_abs + '_vec.tsv'
        out_file_meta_abs = out_file_abs + '_meta.tsv'

        # Write embeddings to file
        print("Writing embeddings to file {}...".format(out_file_vec_abs))
        np.savetxt(out_file_abs + "_vec.tsv", embeddings, delimiter='\t')
        print('Successfully written to file {}.'.format(out_file_vec_abs))

        # Change Unicode code point to character if specified
        if char_as_label:
            codepoints = [chr(int('0x' + codepoint[2:], 16)) for codepoint in codepoints]

        # Write labels
        print("Writing embeddings to file {}...".format(out_file_meta_abs))
        with open("test_meta.tsv", "w+") as f_out:
            for label in codepoints:
                f_out.write(label)
                f_out.write('\n')
        print('Successfully written to file {}.'.format(out_file_meta_abs))


    def __check_create_out_dir_abs(self, out_dir_abs):
        """Check if the given absolute path exists and create if not.

        Args:
            out_dir_abs

        Returns:
            None

        Raises:
            OSError: if specified directory cannot be created
        """
        # Check if out_dir exists and create
        if not os.path.isdir(out_dir_abs):
            print("{} does not exist, creating directory..."
                  .format(out_dir_abs))
            try:
                os.mkdir(out_dir_abs)
            except OSError:
                print("Creation of directory {} failed."
                      .format(out_dir_abs))
                raise
            else:
                print("New directory successfully created.")

