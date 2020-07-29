r"""Visual (image) generation module for confusable detection."""

import os
import random
import shutil
import argparse
from argparse import RawDescriptionHelpFormatter

import qahirah as qah
from qahirah import CAIRO, Colour, Vector

class VisGen:
    """An character image generator for a specific font face.
    
    Initialization:
        >>> vg = VisGen(font_size=28, image_size=36, \
                        font_name="Noto Sans CJK SC", out_dir="test_out")
    Configurations:
        >>> vg.font_name = "Noto Serif CJK SC"
        >>> vg.font_size = 24
        >>> vg.font_style = "SemiBold"
        >>> vg.antialias = "Best"
        >>> vg.grayscale = True
    Visualize single character:
        >>> vg.visualize_single('\u6400')
    Visualize code point in specific range:
        >>> vg.visualize_range(start='\u6400', end='\u64ff')
    Generate dataset from file:
        >>> vg.generate_dataset_from_file('source/dataset.txt', \
        >>>     font_styles=['Bold', 'Regular'], \
        >>>     antialiases=['None', 'Default'])
    Split into train and test dataset:
        >>> vg.train_test_split(num_test=200)
    """

    def __init__(self, font_size=16, image_size=20,
                 font_name="Noto Sans CJK SC", font_style=None,
                 antialias="Default", out_dir="img_out", grayscale=False):
        """Store info about font_size, image_size, out_dir. Search and find
        specified font_name. Set font weight (Regular, Bold, Semi-bold ...).
        Set font anti-aliasing style (Best, Default, Fast...). Set output type.

        Args:
            font_size: Int, size of the font
            image_size: Int, height and width of the output image (in pixel)
            font_name: Str, name of the font face
            font_style: Str, style of the font face (Thin, Bold, Regular...)
            antialias: Str, one of "Default", "Best", "Fast", "Good", "None"
            out_dir: Str, relative path the output directory
            grayscale: Bool, whether to output grayscale or rgb image

        Raises:
            (In setters)
            ValueError: if font_size <= 0 or image_size <= 0
            ValueError: if font_size < image_size
            ValueError: if specified font_name cannot be found
        """
        # Args: font_size, image_size, out_dir
        self.image_size = image_size
        self.font_size = font_size
        self.out_dir = out_dir
        
        # Args: font_name, __ft, __font_face
        # Set freetype library
        self.__ft = qah.get_ft_lib()
        # __font_face is set in the setter for font_name
        self.font_name = font_name

        # Arg: font_style
        self.font_style = font_style

        # Arg: antialias
        self.__antialias_map = {"Default": CAIRO.ANTIALIAS_DEFAULT,
                         "Best": CAIRO.ANTIALIAS_BEST,
                         "Fast": CAIRO.ANTIALIAS_FAST,
                         "Good": CAIRO.ANTIALIAS_GOOD,
                         "None": CAIRO.ANTIALIAS_NONE}
        self.__inv_antialias_map = {v: k for k, v in
                                    self.__antialias_map.items()}
        self.antialias = antialias

        # Arg: grayscale
        self.grayscale = grayscale

        self.__check_out_dir = True


        
    @property
    def font_size(self):
        return self.__font_size
    
    @property
    def image_size(self):
        return self.__image_size

    @property
    def font_name(self):
        return self.__font_name

    @property
    def font_style(self):
        return self.__font_style

    @property
    def antialias(self):
        return self.__inv_antialias_map[self.__antialias]

    @property
    def out_dir(self):
        return self.__out_dir

    @property
    def grayscale(self):
        return self.__grayscale


    @font_size.setter
    def font_size(self, font_size):
        """Check if font_size is 
            1. larger than 0
            2. smaller image_size"""
        if font_size <= 0:
            raise ValueError('Expect font_size to be larger than 0.')
        elif font_size > self.image_size:
            raise ValueError('Expect font_size to be smaller than image_size.')
        else:
            self.__font_size = font_size

    @image_size.setter
    def image_size(self, image_size):
        """Check if image_size is larger than 0."""
        if image_size <= 0:
            raise ValueError('Expect image_size to be larger than 0.')
        else:
            self.__image_size = image_size
    
    @font_name.setter
    def font_name(self, font_name):
        """Check if font_name exists, if so, change to new font_name."""
        if hasattr(self, '__font_name'):
            full_font_name = font_name + ':style=' + self.font_style
        else:
            full_font_name = font_name
        ft_face = self.__ft.find_face(full_font_name) # temporary
        if ft_face.family_name != font_name.split(':')[0]:
            raise ValueError("Font {} cannot be found.".format(font_name))
        else: 
            self.__font_name = font_name
            self.__font_face = qah.FontFace.create_for_ft_face(ft_face)

    @font_style.setter
    def font_style(self, font_style):
        """Add style parameter in in freetype font name. No checking if the
        style exists."""
        # Do nothing if font_style is None or is an empty string
        if not font_style:
            self.__font_style = ""
            return
        # Get freetype font and then create Cairo font
        full_font_name = self.font_name.split(':')[0] + ':style=' + font_style
        ft_face = self.__ft.find_face(full_font_name)  # temporary
        self.__font_style = font_style
        self.__font_face = qah.FontFace.create_for_ft_face(ft_face)

    @antialias.setter
    def antialias(self, antialias):
        if antialias not in self.__antialias_map.keys():
            raise ValueError('Expect antialias to be one of "Default", "Best", '
                             '"Fast", "Good" or "None".')
        self.__antialias = self.__antialias_map[antialias]

    @out_dir.setter
    def out_dir(self, out_dir):
        self.__out_dir = out_dir

    @grayscale.setter
    def grayscale(self, grayscale):
        self.__grayscale = grayscale
        # Configure format and color according to grayscale option
        if grayscale:
            self.__cairo_format = CAIRO.FORMAT_A8 # 8 bits grayscale format
            self.__canvas_color = Colour.grey(0, 0) # grey(i,a) => rgba(i,i,i,a)
            self.__text_color = Colour.grey(0, 1) # Only alpha value matters
        else:
            self.__cairo_format = CAIRO.FORMAT_RGB24
            self.__canvas_color = Colour.grey(255, 1)
            self.__text_color = Colour.grey(0, 1)
            # If /usr/share/X11/rgb.txt exists, color can be specified by
            # Colour.x11['colour_name']

    def generate_dataset_from_file(self, file_path, font_styles, antialiases):
        """Read code points from file and visualize. Character set file must
        follow this format:
            1. Each line represents a single code point
            2. Each code point is in format 'U+2a665'

        Args:
            file_path: Str, path to file for the character set.
            font_styles: list of Str, styles of the font face to visualize
            antialiases: list of Str, antialiasing styles to visualize
        """
        # Check if out_dir exists and create if not
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if self.__check_out_dir:
            self.__check_create_out_dir_abs(out_dir_abs)
        self.__check_out_dir = False

        # Read file and translate format
        code_points = []
        with open(file_path) as f:
            for line in f:
                code_point = chr(int('0x' + line.split('\n')[0][2:], 16))
                code_points.append(code_point)

        self.generate_dataset_from_list(code_points, font_styles, antialiases)


    def generate_dataset_from_list(self, code_points, font_styles, antialiases):
        """Render text images in the given list of code points and write to
        out_dir.

        Args:
            code_points: list of Unicode code points to visualize
            font_styles: list of Str, styles of the font face to visualize
            antialiases: list of Str, antialiasing styles to visualize

        Raises:
            OSError: if specified directory cannot be created
            TypeError: if code point format is not correct
        """
        # Check if out_dir exists and create if not
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if self.__check_out_dir:
            self.__check_create_out_dir_abs(out_dir_abs)
        self.__check_out_dir = False

        for font_style in font_styles:
            self.font_style = font_style
            print('Successfully selected font style: {}.'.format(font_style))
            for antialias in antialiases:
                self.antialias = antialias
                print('Successfully selected antialiasing style: {}.'.format(
                    antialias))
                self.visualize_list(code_points)

        self.__check_out_dir = True


    def visualize_list(self, code_points, x=None, y=None):
        """Render text images from start code point to end code point and write
        to out_dir.

        Args:
            code_points: List of Str, a list of all codepoints to visualize
            x: Int, x coordinate of the position of text, in number of pixels
            y: Int, y coordinate of the position of text, in number of pixels

        Raises:
            OSError: if specified directory cannot be created
            TypeError: if code point format is not correct
        """
        # Check if out_dir exists and create if not
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if self.__check_out_dir:
            self.__check_create_out_dir_abs(out_dir_abs)
        self.__check_out_dir = False

        # Visualize list of code points
        print("Visualizing {} total code points.".format(len(code_points)))
        for idx, code_point in enumerate(code_points):
            if idx % 50 == 0:
                print("Now writing {}st code point.".format(idx + 1))
            self.visualize_single(code_point, False, x=x, y=y)
        print("Finished.")
        print("Images stored in directory {}.".format(out_dir_abs))

        self.__check_out_dir = True


    def visualize_range(self, start, end, x=None, y=None):
        """Render text images from start code point to end code point and write
        to out_dir.
        
        Args:
            start: Str, Unicode code point, starting code point to write
            end: Str, Unicode code point, the last code point to write
            x: Int, x coordinate of the position of text, in number of pixels
            y: Int, y coordinate of the position of text, in number of pixels

        Raises:
            OSError: if specified directory cannot be created
            TypeError: if start or end is not a single character code point
        """
        # Check if out_dir exists and create if not
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if self.__check_out_dir:
            self.__check_create_out_dir_abs(out_dir_abs)
        self.__check_out_dir = False

        # Compute ord for start and end code points
        try:
            ord_start = ord(start)
            ord_end = ord(end)
        except TypeError:
            print("Expect start and end to be single character code point.")
            raise

        # Check start and end ord and exchange if necessary
        if ord_start > ord_end:
            ord_start, ord_end = ord_end, ord_start
        
        # Visualize all code points
        total_cp = ord_end - ord_start + 1
        print("Visualizing {} total code points from {} to {}."
              .format(total_cp, start, end))
        for cur_ord in range(ord_start, ord_end+1):
            if (cur_ord - ord_start) % 50 == 0:
                print("Now writing {}st code point."
                      .format(str(cur_ord - ord_start + 1)))
            code_point = chr(cur_ord)
            self.visualize_single(code_point, False, x=x, y=y)
        print("Finished.")
        print("Images stored in directory {}.".format(out_dir_abs))

        self.__check_out_dir = True


    def visualize_single(self, code_point, check_out_dir=True, x=None, y=None):
        """Write rendered text image for specific code point to output
        directory, text is positioned at (x, y). If both x and y are set to None
        (default), the text will be centered.

        Args:
            code_point: Str, single unicode code point
            check_out_dir: Bool, set True if need to check output directory
            x: Int, x coordinate of the position of text, in number of pixels.
            y: Int, y coordinate of the position of text, in number of pixels.

        Returns:
            file_path: Str, path to file

        Raises:
            OSError: if specified directory cannot be created
        """
        # Get absolute directory path
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if check_out_dir:
            self.__check_create_out_dir_abs(out_dir_abs)

        # Create and configure ImageSurface
        figure_dimensions = Vector(self.image_size, self.image_size)
        pix = qah.ImageSurface.create(format=self.__cairo_format,
                                      dimensions=figure_dimensions)

        # Create context
        ctx = qah.Context.create(pix)

        # Select anti-aliasing style in font options
        font_options = ctx.font_options
        font_options.antialias = self.__antialias
        # Set color
        ctx.set_source_colour(self.__canvas_color)
        ctx.paint()
        ctx.set_source_colour(self.__text_color)
        # Set font face and size
        ctx.set_font_face(self.__font_face)
        ctx.set_font_size(self.__font_size)
        ctx.set_font_options(font_options)

        # Position text
        if x is None and y is None:
            self.__center_text(ctx, code_point)
        elif x is None or y is None:
            raise ValueError("Expect both x and y to be specified for "
                             "alternative positioning.")
        else:
            self.__position_text(ctx, code_point, x, y)

        # Get file name and file path
        filename = self.__get_filename(code_point)
        file_path = os.path.join(out_dir_abs, filename)

        # Show text and write to file
        ctx.show_text(code_point)
        pix.flush()
        pix.write_to_png(file_path)

        return file_path

    def train_test_split(self, num_test=100):
        """Split dataset (already created) into training and testing dataset.
        The number of test records needs to be specified.

        Args:
            num_test: Int, number of test records

        Returns:
            num_train: Int, total number of training records
            num_test: Int, total number of test records

        Raises:

        """
        # Get absolute path to train and test data directory
        train_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        test_dir_abs = os.path.join(os.getcwd(), self.out_dir + '_test')

        # Create test dir
        self.__check_create_out_dir_abs(test_dir_abs)

        # Get total number of training records
        num_total = len([name for name in os.listdir(train_dir_abs)
                         if os.path.isfile(os.path.join(train_dir_abs, name))])
        num_exist = len([name for name in os.listdir(test_dir_abs)
                         if os.path.isfile(os.path.join(test_dir_abs, name))])
        if num_total == 0:
            raise OSError('No data found in specified out_dir.')
        if num_test > num_total:
            raise ValueError('Expect num_test to be smaller than total number '
                             'of records.')
        if num_exist != 0:
            raise OSError('Test data already exists.')
        num_train = num_total - num_test

        # Do train/test split
        print('Creating train test split with {} total records...'
              .format(num_total))
        print('Train size: {}'.format(num_train))
        print('Test size: {}'.format(num_test))
        filenames = random.sample(os.listdir(train_dir_abs), 100)
        for filename in  filenames:
            srcpath = os.path.join(train_dir_abs, filename)
            shutil.move(srcpath, test_dir_abs)
        print('Train test split successfully created.')

        # Check number of classes in each split
        class_train = set([name.split('_')[0] for name in
                           os.listdir(train_dir_abs)])
        class_test = set([name.split('_')[0] for name in
                          os.listdir(test_dir_abs)])
        no_missing_class = class_test.issubset(class_train)
        print('Training dataset has {} categories.'.format(len(class_train)))
        print('Test dataset has {} categories.'.format(len(class_test)))
        print('All test categories in training data: {}'
              .format(no_missing_class))

        return num_train, num_test



    def __get_filename(self, code_point):
        """Get the filename for code point under current context. Filename is
        'CODEPOINT_FONTNAME_FONTSTYLE_ANTIALIAS'

        Args:
            code_point: Single Unicode code point

        Returns:
            filename: Str, corresponding file name
        """
        filename = 'U+' + str(hex(ord(code_point)))[2:]
        filename += '_'
        filename += self.__font_name
        filename += '_'
        if self.__font_style:
            filename += self.font_style
            filename += '_'
        filename += self.antialias
        filename += '.png'
        return filename


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

    def __position_text(self, context, text, x, y):
        """Put text in the intended position (relative to the origin)."
        
        Args:
            context: Qahirah context for the selected font and text size
            text: Unicode codepoint for the text (character)
            x: X coordinate for the intended position
            y: Y coordinate for the intended position
        """
        # Calculate how much to move on x and y axis
        text_extents = context.text_extents(text)
        delta_x = x - text_extents.x_bearing
        delta_y = y - text_extents.y_bearing

        # Move in context
        context.move_to((delta_x, delta_y))

    def __center_text(self, context, text):
        """Put text in the center of the image (vertically and horizontally).
    
        Args:
            context: Qahirah context for the selected font and text size
            text: Unicode code point for the text (character)
        """
    
        # Calculate how much to move on x and y axis
        text_extents = context.text_extents(text)
        text_height = text_extents.height
        text_width = text_extents.width
        pos_x = (self.image_size - text_width) / 2
        pos_y = (self.image_size - text_height) / 2
        delta_x = pos_x - text_extents.x_bearing
        delta_y = pos_y - text_extents.y_bearing
        
        # Move in context
        context.move_to((delta_x, delta_y))
    

if __name__ == "__main__":
    formatter = RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description='Usage: \n',
                                     formatter_class=formatter)
    parser.add_argument('--font_size', type=int, default=36, required=False,
                        nargs=1, help='Font size.')
    parser.add_argument('--image_size', type=int, default=40, required=False,
                        nargs=1, help='Image height and width (in pixels).')
    parser.add_argument('--font_name', type=str, default="Noto Sans CJK SC",
                        required=False, nargs=1, help="Font name.")
    parser.add_argument('--out_dir', type=str, default="img_out",
                        required=False, nargs=1,
                        help="Relative path to output directory.")
    parser.add_argument('--code_point_range', type=str, required=True, nargs=2,
                        help="Start and end of the range of code points to "
                             "visualize.")
    parser.add_argument('--grayscale', type=bool, required=False, nargs=1,
                        help="Whether to output grayscale or rgb image.")
    args = parser.parse_args()
    
    vg = VisGen(font_size=args.font_size, image_size=args.image_size,
                font_name=args.font_name, out_dir=args.out_dir)
    vg.visualize_range(args.code_point_range[0], args.code_point_range[1])


"""TODO: - Configure root directory for project.
          - Write to buffer instead of file.
          - Tofu and blank detection."""
