r"""Visual (image) generation module for confusable detection.

To use vis_gen.py as a command-line tool:
$ python3 vis_gen.py --font_size 32 --image_size 40 \
    --font_name "Noto Sans CJK TC" --code_point_range '\u6400', '\u64ff'
"""

import os
import argparse
from argparse import RawDescriptionHelpFormatter

import qahirah as qah
from qahirah import CAIRO, Colour, Vector

class VisGen:
    """An character image generator for a specific font face.
    
    To use:
        >>> vg = VisGen(font_size=28, image_size=36, \
                        font_name="Noto Sans CJK SC", out_dir="test_out")
        >>> vg.visualize_range(start='\u6400', end='\u64ff')
        >>> vg.font_name = "Noto Serif CJK SC"
        >>> vg.font_size = 24
        >>> vg.grayscale = True
        >>> vg.visualize_single('\u6400')

    """

    def __init__(self, font_size=36, image_size=40,
                 font_name="Noto Sans CJK SC", out_dir="img_out",
                 grayscale=False):
        """Store info about font_size, image_size, out_dir.
        Search and find specified font_name.

        Args:
            font_size: Int, size of the font
            image_size: Int, height and width of the output image (in pixel)
            font_face: Str, name of the font face
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

        # Arg: grayscale
        self.grayscale = grayscale

        
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
        ft_face = self.__ft.find_face(font_name) # temporary
        if ft_face.family_name != font_name:
            raise ValueError("Font {} cannot be found.".format(font_name))
        else: 
            self.__font_name = font_name
            self.__font_face = qah.FontFace.create_for_ft_face(ft_face)
    
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
        if not os.path.isdir(out_dir_abs):
            print("{} does not exist, creating directory.".format(out_dir_abs))
            try: 
                os.mkdir(out_dir_abs)
            except OSError:
                print("Creation of directory {} failed.".format(out_dir_abs))
                raise
            else:
                print("New directory successfully created.")

        # Check if out_dir/font_name exists and create if not
        font_dir_abs = os.path.join(os.getcwd(), self.out_dir,
                                    self.font_name.replace(" ", "_"))
        if not os.path.isdir(font_dir_abs):
            print("{} does not exist, creating directory.".format(font_dir_abs))
            try:
                os.mkdir(font_dir_abs)
            except OSError:
                print("Creation of directory {} failed.".format(font_dir_abs))
                raise
            else:
                print("New directory successfully created")

        # Compute ord for start and end code points
        try:
            ord_start = ord(start)
            ord_end = ord(end)
        except:
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
            if (cur_ord - ord_start) % 16 == 0:
                print("Now writing {}st code point."
                      .format(str(cur_ord - ord_start + 1)))
            code_point = chr(cur_ord)
            self.visualize_single(code_point, check_out_dir=False, x=x, y=y)
        print("Finished.")
        print("Images stored in directory {}.".format(font_dir_abs))

    def visualize_single(self, code_point, check_out_dir=True, x=None,
                            y=None):
        """Write rendered text image for specific code point to output
        directory, text is positioned at (x, y). If both x and y are set to None
        (default), the text will be centered.

        Args:
            code_point: Str, single unicode code point
            check_out_dir: Bool, set True if need to check output directory
            x: Int, x coordinate of the position of text, in number of pixels.
            y: Int, y coordinate of the position of text, in number of pixels.
            
        Raises:
            OSError: if specified directory cannot be created
        """
        # font_dir_abs: absolute path to output directory
        font_dir_abs = os.path.join(os.getcwd(), self.out_dir,
                                    self.font_name.replace(" ", "_"))
        if check_out_dir:
            # Check if out_dir exists and create
            out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
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
            
            if not os.path.isdir(font_dir_abs):
               print("{} does not exist, creating directory..."
                     .format(font_dir_abs))
               try:
                   os.mkdir(font_dir_abs)
               except OSError:
                   print("Creation of directory {} failed."
                         .format(font_dir_abs))
                   raise
               else:
                   print("New directory successfully created")

        # Create and configure ImageSurface
        figure_dimensions = Vector(self.image_size, self.image_size)
        pix = qah.ImageSurface.create(format=self.__cairo_format,
                                      dimensions=figure_dimensions)

        # Create context
        ctx = qah.Context.create(pix)
        ctx.set_source_colour(self.__canvas_color)
        ctx.paint()
        ctx.set_source_colour(self.__text_color)
        ctx.set_font_face(self.__font_face)
        ctx.set_font_size(self.font_size)

        # Position text
        if x is None and y is None:
            self.center_text(ctx, code_point)
        elif x is None or y is None:
            raise ValueError("Expect both x and y to be specified for "
                             "alternative positioning.")
        else:
            self.position_text(ctx, code_point, x, y)

        # Show text and write to file
        ctx.show_text(code_point)
        pix.flush()
        pix.write_to_png(os.path.join(font_dir_abs, str(ord(code_point)) +
                                      ".png"))

    def position_text(self, context, text, x, y):
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

    def center_text(self, context, text):
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
