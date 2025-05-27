# Copyright (C) 2020 and later: Unicode, Inc. and others.
# License & terms of use: http://www.unicode.org/copyright.html

import argparse
from argparse import RawDescriptionHelpFormatter
import os
import qahirah as qah
from qahirah import CAIRO, Colour, Vector

class VisualGenerator:
    """An character image generator for a specific font face. Also serve as
    dataset generator.
    
    Initialization:
        >>> vg = VisualGenerator(font_size=28, image_size=36, \
                        font_name="Noto Sans CJK SC", out_dir="test_out")

    Configurations:
        >>> vg.font_name = "Noto Serif CJK SC"
        >>> vg.font_size = 24
        >>> vg.font_style = "SemiBold"
        >>> vg.antialias = "Best"
        >>> vg.grayscale = True

    Visualize single character:
    - In the directory 'out_dir', Use the configuration in vg to generate image
        of a single Unicode character.
        >>> vg.visualize_single('\u6400')

    Visualize code point in specific range:
    - In the directory 'out_dir', use the configuration in vg to generate images
        of multiple Unicode characters with starting and ending code points as
        specified.
        >>> vg.visualize_range(start='\u6400', end='\u64ff')

    Generate dataset from file:
    - In the directory 'out_dir', according to configuration, generate images of
        all the code points in specified file. Each code point will generate
        multiple images for every font style and anti-aliasing style. In the
        source file, each code point should be in format 'U+XXXX' and code
        points should be separated by new line.
        >>> vg.generate_dataset_from_file('source/dataset.txt', \
        >>>     font_styles=['Bold', 'Regular'], \
        >>>     antialiases=['None', 'Default'])

    Split images in out_dir into train and test dataset:
    - From directory 'out_dir', randomly select and move N images to new
        directory named 'out_dir_test'. N is specified by num_test.
        >>> vg.train_test_split(num_test=200)
    """

    def __init__(self, font_size=16, image_size=20,
                 font_name="Noto Sans CJK SC", font_style=None,
                 antialias="Default", out_dir="data", grayscale=False):
        """Store info about font_size, image_size, out_dir. Search and find
        specified font_name. Set font weight (Regular, Bold, Semi-bold ...).
        Set font anti-aliasing style (Best, Default, Fast...). Set output type.

        Args:
            font_size: Int or Float, size of the font (default 16).
            image_size: Int or Float, height and width of the output image (in
                pixel) (default 20).
            font_name: Str, name of the font face (default "Noto Sans CJK SC").
            font_style: Str, style of the font face (None, "Thin", "Bold",
                "Regular"...) (default None). IMPORTANT: If font_style does not
                exists, no exception will be raise and font style will default
                to 'Regular'.
            antialias: Str, one of "Default", "Best", "Fast", "Good", "None"
                (default "Default").
            out_dir: Str, relative path of the output directory (default
                "data").
            grayscale: Bool, whether to output grayscale or rgb image (default
                False).

        Raises:
            (In setters)
            ValueError: if font_size <= 0 or image_size <= 0.
            ValueError: if font_size < image_size.
            ValueError: if specified font_name cannot be found.
        """
        # Args: font_size, image_size, out_dir
        self.image_size = image_size
        self.font_size = font_size
        self.out_dir = out_dir
        
        # Args: font_name, _ft, _font_face
        # Set freetype library
        self._ft = qah.get_ft_lib()
        # _font_face is set in the setter for font_name
        self.font_name = font_name

        # Arg: font_style
        self.font_style = font_style

        # Arg: antialias
        # self._antialias_map: dictionary mapping anti-aliasing style string to
        #   Cairo antiliasing style enumeration class.
        # self._inv_antialias_map: inverse mapping of self._antialias_map
        self._antialias_map = {"Default": CAIRO.ANTIALIAS_DEFAULT,
                         "Best": CAIRO.ANTIALIAS_BEST,
                         "Fast": CAIRO.ANTIALIAS_FAST,
                         "Good": CAIRO.ANTIALIAS_GOOD,
                         "None": CAIRO.ANTIALIAS_NONE}
        self._inv_antialias_map = {v: k for k, v in
                                    self._antialias_map.items()}
        self.antialias = antialias

        # Arg: grayscale
        self.grayscale = grayscale

        # Set attribute _check_out_dir is a flag to avoid multiple checks on
        # the existence of self.out_dir. For example, when function
        # visualize_range() calls visualize_single() repeatedly, there is no
        # need to repeated check the existence of out_dir in every call.
        self._check_out_dir = True


        
    @property
    def font_size(self):
        """
        Returns:
             self._font_size: Int or Float, positive number indicating font
                size.
        """
        return self._font_size
    
    @property
    def image_size(self):
        """
        Returns:
             self._image_size: Int or Float, positive number indicating image
                size. This value is always larger than font_size.
        """
        return self._image_size

    @property
    def font_name(self):
        """
        Returns:
             self._font_name: Str, name of the font face.
        """
        return self._font_name

    @property
    def font_style(self):
        """
        Returns:
            self._font_style: Str, style of the font face ("Thin", "Bold",
                "Regular"...). If not font_style set by user, returns empty
                string "".
        """
        return self._font_style

    @property
    def antialias(self):
        """
        Returns:
             antialias: Str, one of "Default", "Best", "Fast", "Good", "None".
        """
        return self._inv_antialias_map[self._antialias]

    @property
    def out_dir(self):
        """
        Returns:
            self._out_dir: Str, relative path of the output directory.
        """
        return self._out_dir

    @property
    def grayscale(self):
        """
        Returns:
            self._grayscale: Bool, true if format of the output is in garyscale.
        """
        return self._grayscale


    @font_size.setter
    def font_size(self, font_size):
        """Raises ValueError if:
            1. font_size is smaller or equal to 0
            2. font_size is larger than image_size."""
        if hasattr(self, '_image_size') and font_size > self.image_size:
            raise ValueError('Expect font_size to be smaller than image_size.')

        if font_size <= 0:
            raise ValueError('Expect font_size to be larger than 0.')
        else:
            self._font_size = font_size

    @image_size.setter
    def image_size(self, image_size):
        """Raises ValueError if:
            1. image_size is smaller or equal to 0
            2. image_size is smaller than font_size."""
        if hasattr(self, '_font_size') and image_size < self.font_size:
            raise ValueError('Expect image_size to be larger than font_size.')

        if image_size <= 0:
            raise ValueError('Expect image_size to be larger than 0.')
        else:
            self._image_size = image_size
    
    @font_name.setter
    def font_name(self, font_name):
        """Check if font_name exists in system. If so, change self._font_name
        to new font_name and set self._font_face as the corresponding Qahirah
        font face. If font_name does not exists in system, raise ValueError."""
        if hasattr(self, '_font_name') and self._font_style:
            # If the attribute _font_name is set, object is initialized. Make
            # sure font_style is also included for free type font.
            full_font_name = font_name + ':style=' + self.font_style
        else:
            # If _font_name is not set, this is the initialization process.
            # Don't worry about font_style since it will be added later.
            full_font_name = font_name
        # Get temporary font free type font face
        ft_face = self._ft.find_face(full_font_name)

        # If font name exists, set _font_name and get Qahirah font face
        if ft_face.family_name != font_name.split(':')[0]:
            raise ValueError("Font {} cannot be found.".format(font_name))
        else: 
            self._font_name = font_name
            self._font_face = qah.FontFace.create_for_ft_face(ft_face)

    @font_style.setter
    def font_style(self, font_style):
        """Add style parameter in in freetype font name and get corresponding
        Qahirah font face. Both self._font_style and self._font_face will be
        changed. IMPORTANT: If font_style does not exists in system, font_style
        configuration will be ignored and regular style will be selected.
        TODO: Check if font_style exists in system."""
        # Do nothing if font_style is None or is an empty string
        if not font_style:
            full_font_name = self.font_name.split(':')[0]
            ft_face = self._ft.find_face(full_font_name)  # temporary
        else:
            # Get freetype font and then create Cairo font
            full_font_name = self.font_name.split(':')[0] + ':style=' + \
                             font_style
            ft_face = self._ft.find_face(full_font_name)  # temporary

        # Create and store new font face
        self._font_style = font_style if font_style else ""
        self._font_face = qah.FontFace.create_for_ft_face(ft_face)

    @antialias.setter
    def antialias(self, antialias):
        """If antialias style exists in Qahirah anti-aliasing styles, set
        self._antialias as corresponding CAIRO antiliasing style enumeration
        class."""
        if antialias not in self._antialias_map.keys():
            raise ValueError('Expect antialias to be one of "Default", "Best", '
                             '"Fast", "Good" or "None".')
        self._antialias = self._antialias_map[antialias]

    @out_dir.setter
    def out_dir(self, out_dir):
        """Set relative path to output directory."""
        self._out_dir = out_dir

    @grayscale.setter
    def grayscale(self, grayscale):
        """Set self._graycale to boolean value indicating whether to format
        output to grayscale.
        If grayscale is true, set self._cairo_format as CAIRO.FORMAT_A8. Set
        self._canvas_color as CAIRO colour black. Set alpha value of
        self._text_color to 1 (no transparency).
        If grayscale is false, set self._cairo_format as CAIRO.FORMAT_RGB24 (24
        bit RGB format). Set self._canvas_color as CAIRO colour white and text
        color as black."""
        self._grayscale = grayscale
        # Configure format and color according to grayscale option
        if grayscale:
            self._cairo_format = CAIRO.FORMAT_A8 # 8 bits grayscale format
            self._canvas_color = Colour.grey(0, 0) # grey(i,a) => rgba(i,i,i,a)
            self._text_color = Colour.grey(0, 1) # Only alpha value matters
        else:
            self._cairo_format = CAIRO.FORMAT_RGB24
            self._canvas_color = Colour.grey(255, 1)
            self._text_color = Colour.grey(0, 1)
            # If /usr/share/X11/rgb.txt exists, color can be specified by
            # Colour.x11['colour_name']

    def generate_dataset_from_file(self, file_path, font_styles, antialiases):
        """Read code points from file and visualize. Character set file must
        follow this format:
            1. Each line represents a single code point.
            2. Each code point is in format 'U+2a665'.
        Example file content:
            U+4E00
            U+4EDA
            U+5231
            ...
            U+6533

        Args:
            file_path: Str, path to file for the character set.
            font_styles: list of Str, styles of the font face to visualize.
            antialiases: list of Str, antialiasing styles to visualize.
        """
        # Check if out_dir exists and create if not
        out_dir_abs = self._get_out_dir_abs_and_check()

        # Read file and translate format
        code_points = []
        with open(file_path) as f:
            for line in f:
                code_point = chr(int('0x' + line.split('\n')[0][2:], 16))
                code_points.append(code_point)

        self.generate_dataset_from_list(code_points, font_styles, antialiases)

        # Flag flipped to False in self._get_out_dir_abs_and_check
        self._check_out_dir = True


    def generate_dataset_from_list(self, code_points, font_styles, antialiases):
        """Render text images in the given list of code points and write to
        out_dir.

        Args:
            code_points: list of Unicode code points to visualize.
            font_styles: list of Str, styles of the font face to visualize.
            antialiases: list of Str, antialiasing styles to visualize.

        Raises:
            OSError: if specified directory cannot be created.
            TypeError: if code point format is not correct.
        """
        # Check if out_dir exists and create if not
        out_dir_abs = self._get_out_dir_abs_and_check()

        for font_style in font_styles:
            self.font_style = font_style
            print('Successfully selected font style: {}.'.format(font_style))
            for antialias in antialiases:
                self.antialias = antialias
                print('Successfully selected antialiasing style: {}.'.format(
                    antialias))
                self.visualize_list(code_points)

        # Flag flipped to False in self._get_out_dir_abs_and_check
        self._check_out_dir = True


    def visualize_list(self, code_points, x=None, y=None):
        """Render text images from start code point to end code point and write
        to out_dir.

        Args:
            code_points: List of Str, a list of all codepoints to visualize.
            x: Int, x coordinate of the position of text, in number of pixels.
            y: Int, y coordinate of the position of text, in number of pixels.

        Raises:
            OSError: if specified directory cannot be created.
            TypeError: if code point format is not correct.
        """
        # Check if out_dir exists and create if not
        out_dir_abs = self._get_out_dir_abs_and_check()

        # Visualize list of code points
        print("Visualizing {} total code points.".format(len(code_points)))
        for idx, code_point in enumerate(code_points):
            if idx % 50 == 0:
                print("Now writing {}st code point.".format(idx + 1))
            self.visualize_single(code_point, False, x=x, y=y)
        print("Finished.")
        print("Images stored in directory {}.".format(out_dir_abs))

        # Flag flipped to False in self._get_out_dir_abs_and_check
        self._check_out_dir = True


    def visualize_range(self, start, end, x=None, y=None):
        """Render text images from start code point to end code point and write
        to out_dir.
        
        Args:
            start: Str, Unicode code point, starting code point to write.
            end: Str, Unicode code point, the last code point to write.
            x: Int, x coordinate of the position of text, in number of pixels.
            y: Int, y coordinate of the position of text, in number of pixels.

        Raises:
            OSError: if specified directory cannot be created.
            TypeError: if start or end is not a single character code point.
            ValueError: if start comes after end.
        """
        # Check if out_dir exists and create if not
        out_dir_abs = self._get_out_dir_abs_and_check()

        # Compute ord for start and end code points
        try:
            ord_start = ord(start)
            ord_end = ord(end)
        except TypeError:
            print("Expect start and end to be single character code point.")
            raise

        # Check start and end ord in order
        if ord_start > ord_end:
            raise ValueError('Expect ord_start ot be smaller than ord_end.')
        
        # Visualize all code points
        total_cp = ord_end - ord_start + 1
        print("Visualizing {} total code points from U+{:04X} to U+{:04X}."
              .format(total_cp, ord(start), ord(end)))
        for cur_ord in range(ord_start, ord_end + 1):
            if (cur_ord - ord_start) % 50 == 0:
                print("Now writing {}st code point."
                      .format(str(cur_ord - ord_start + 1)))
            code_point = chr(cur_ord)
            self.visualize_single(code_point, False, x=x, y=y)
        print("Finished.")
        print("Images stored in directory {}.".format(out_dir_abs))

        # Flag flipped to False in self._get_out_dir_abs_and_check
        self._check_out_dir = True


    def visualize_single(self, code_point, check_out_dir=True, x=None, y=None):
        """Write rendered text image for specific code point to output
        directory, text is positioned at (x, y). If both x and y are set to None
        (default), the text will be centered.

        Args:
            code_point: Str, single unicode code point.
            check_out_dir: Bool, set True if need to check output directory.
            x: Int, x coordinate of the position of text, in number of pixels.
            y: Int, y coordinate of the position of text, in number of pixels.

        Returns:
            file_path: Str, path to file. The filename of .png

        Raises:
            OSError: if specified directory cannot be created.
        """
        # Get absolute directory path
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if check_out_dir:
            self._check_create_out_dir_abs(out_dir_abs)

        # Create and configure ImageSurface
        figure_dimensions = Vector(self.image_size, self.image_size)
        surface = qah.ImageSurface.create(format=self._cairo_format,
                                      dimensions=figure_dimensions)

        # Create context
        ctx = qah.Context.create(surface)

        # Select anti-aliasing style in font options
        font_options = ctx.font_options
        font_options.antialias = self._antialias
        # Set color
        ctx.set_source_colour(self._canvas_color)
        # Paints background
        ctx.paint()
        ctx.set_source_colour(self._text_color)
        # Set font face and size
        ctx.set_font_face(self._font_face)
        ctx.set_font_size(self._font_size)
        ctx.set_font_options(font_options)

        # Position text
        if x is None and y is None:
            self._center_text(ctx, code_point)
        elif x is None or y is None:
            raise ValueError("Expect both x and y to be specified for "
                             "alternative positioning.")
        else:
            self._position_text(ctx, code_point, x, y)

        # Show text, flush ImageSurface
        ctx.show_text(code_point)
        surface.flush()

        # Write to file
        filename = self._get_filename(code_point)
        file_path = os.path.join(out_dir_abs, filename)
        surface.write_to_png(file_path)

        return file_path

    def _get_filename(self, code_point):
        """Get the filename for code point under current context. Filename is
        'CODEPOINT_FONTNAME[_FONTSTYLE]_ANTIALIAS.png'

        Example file names:
        U+4e05_Noto Sans CJK SC_SemiBold_Default.png
        U+6bb5_Noto Sans CJK TC_DemiLight_Default.png
        U+2f25_Noto Sans CJK SC_Default.png
        U+6ad9_Noto Sans CJK SC_Default.png

        Args:
            code_point: Single Unicode code point.

        Returns:
            filename: Str, corresponding file name.
        """
        filename = 'U+{:04X}'.format(ord(code_point))
        filename += '_'
        filename += self._font_name
        filename += '_'
        if self._font_style:
            filename += self.font_style
            filename += '_'
        filename += self.antialias
        filename += '.png'
        return filename

    def _get_out_dir_abs_and_check(self):
        """Get absolute path to out_dir and check the existence of such
        directory. Create new directory if necessary.

        Returns:
            out_dir_abs: Str, absolute path to the out_dir.
        """
        # Check if out_dir exists and create if not
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if self._check_out_dir:
            self._check_create_out_dir_abs(out_dir_abs)
        self._check_out_dir = False

        return out_dir_abs

    def _check_create_out_dir_abs(self, out_dir_abs):
        """Check if the given absolute path exists and create if not.

        Args:
            out_dir_abs: Str, absolute path to output directory.

        Raises:
            OSError: if specified directory cannot be created.
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

    def _position_text(self, context, text, x, y):
        """Put text in the intended position (relative to the origin)."
        
        Args:
            context: Qahirah context for the selected font and text size.
            text: Unicode codepoint for the text (character).
            x: X coordinate for the intended position.
            y: Y coordinate for the intended position.
        """
        # Calculate how much to move on x and y axis
        text_extents = context.text_extents(text)
        delta_x = x - text_extents.x_bearing
        delta_y = y - text_extents.y_bearing

        # Move in context
        context.move_to((delta_x, delta_y))

    def _center_text(self, context, text):
        """Put text in the center of the image (vertically and horizontally).
    
        Args:
            context: Qahirah context for the selected font and text size.
            text: Unicode code point for the text (character).
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
    parser.add_argument('--out_dir', type=str, default="data",
                        required=False, nargs=1,
                        help="Relative path to output directory.")
    parser.add_argument('--code_point_range', type=str, required=True, nargs=2,
                        help="Start and end of the range of code points to "
                             "visualize.")
    parser.add_argument('--grayscale', type=bool, required=False, nargs=1,
                        help="Whether to output grayscale or rgb image.")
    args = parser.parse_args()
    
    vg = VisualGenerator(font_size=args.font_size, image_size=args.image_size,
                font_name=args.font_name, out_dir=args.out_dir)
    vg.visualize_range(args.code_point_range[0], args.code_point_range[1])


"""TODO: - Configure root directory for project.
          - Write to buffer instead of file.
          - Tofu and blank detection."""
