r"""Visual (image) generation module for confusable detection."""
"""TODOS: - Configure root directory for project.
          - Write to buffer instead of file.
          - Grayscale image.
          - Tofu and blank detection."""

import sys
import os
import math
import argparse
from argparse import RawDescriptionHelpFormatter

import qahirah as qah
from qahirah import CAIRO, Colour, Vector

class VisGen:
    """An character image generator for a specific font face.
    
    To use:
    >>> vg = VisGen(font_size = 28, image_size=36, font_name="")
    """

    def __init__(self, font_size=36, image_size=40, font_name="Noto Sans CJK SC", out_dir="img_out"):
        """Store info about font_size, image_size, out_dir. Search and find specified font_name.
        Assertion:
            1. both font_size and image_size are larger than 0
            2. font_size is smaller than image_size
            3. font_face exists in the freetype library

        Args:
            font_size: Int, size of the font
            image_size: Int, height and width of the output image (in pixel)
            font_face: Str, name of the font face
            out_dir: Str, name of the output directory, relative to the root directory.

        Returns:
            None
        """
        # Args: font_size, image_size, out_dir
        # Check font_size and image_size
        if font_size <= 0 or image_size <= 0:
            raise ValueError('Expect both font_size and image_size to be larger than 0.')
        # Check that image_size is larger than font_size
        if font_size > image_size:
            raise ValueError('Expect font_size to be smaller than image_size.')
        # Set properties
        self.image_size = image_size
        self.font_size = font_size
        self.out_dir = out_dir
        
        # Args: font_name, __ft, __font_face
        # Find and check freetype font face
        self.__ft = qah.get_ft_lib()
        ft_face = self.__ft.find_face(font_name) # temporarily used for creating cairo face
        if ft_face.family_name != font_name:
            raise ValueError("Specified font {} cannot be found.".format(font_name))
        # Set property
        self.font_name = font_name
        self.__font_face = qah.FontFace.create_for_ft_face(ft_face)
        
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
        ft_face = self.__ft.find_face(font_name) # temporarily used for creating cairo face
        if ft_face.family_name != font_name:
            raise ValueError("Specified font {} cannot be found.".format(font_name))
        else: 
            self.__font_name = font_name
            self.__font_face = qah.FontFace.create_for_ft_face(ft_face)
    
    @out_dir.setter
    def out_dir(self, out_dir):
        self.__out_dir = out_dir
    
    def write_range(self, start, end):
        """Write rendered text images from start code point to end code point.
        
        Args:
            start: Unicode code point, starting code point to write
            end: Unicode code point, the last code point to write

        Return: 
        """
        # Check if out_dir exists and create
        out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
        if (not os.path.isdir(out_dir_abs)):
            print("{} does not exist, creating directory...".format(out_dir_abs))
            try: 
                os.mkdir(out_dir_abs)
            except OSError:
                print("Creation of directory {} failed.".format(out_dir_abs))
                raise
            else:
                print("New directory successfully created.")
        
        font_dir_abs = os.path.join(os.getcwd(), self.out_dir, self.font_name.replace(" ", "_"))
        if (not os.path.isdir(font_dir_abs)):
           print("{} does not exist, creating directory...".format(font_dir_abs))
           try:
               os.mkdir(font_dir_abs)
           except OSError:
               print("Creation of directory {} failed.".format(font_dir_abs))
               raise
           else:
               print("New directory successfully created")

        # Compute ord for start and end code points
        ord_start = ord(start)
        ord_end = ord(end)

        # Check start and end ord and exchange if necessary
        if ord_start > ord_end:
            ord_start, ord_end = ord_end, ord_start
        
        # Visualize all code points
        total_cp = ord_end - ord_start + 1
        print("Visualizing {} total code points from {} to {}.".format(total_cp, start, end))
        for cur_ord in range(ord_start, ord_end+1):
            if (cur_ord - ord_start) % 16 == 0:
                print("Now writing {}st code point.".format(str(cur_ord - ord_start + 1)))
            code_point = chr(cur_ord)
            self.write_visualization_center(code_point, check_out_dir=False)
        print("Finished.")
        print("Images stored in directory {}.".format(font_dir_abs))

        

    def write_visualization_center(self, code_point, check_out_dir=True):
        """Write rendered text image for specific code point to output directory.
            1. Check if out_dir exists and create if needed
            2. Center text
            3. Write to out_dir

        Args:
            code_pint: single unicode code point
            check_out_dir: wether or not to check if output directory exists
            
        Returns:
            None
        """
        font_dir_abs = os.path.join(os.getcwd(), self.out_dir, self.font_name.replace(" ", "_"))
        if check_out_dir:
            # Check if out_dir exists and create
            out_dir_abs = os.path.join(os.getcwd(), self.out_dir)
            if (not os.path.isdir(out_dir_abs)):
                print("{} does not exist, creating directory...".format(out_dir_abs))
                try: 
                    os.mkdir(out_dir_abs)
                except OSError:
                    print("Creation of directory {} failed.".format(out_dir_abs))
                    raise
                else:
                    print("New directory successfully created.")
            
            if (not os.path.isdir(font_dir_abs)):
               print("{} does not exist, creating directory...".format(font_dir_abs))
               try:
                   os.mkdir(font_dir_abs)
               except OSError:
                   print("Creation of directory {} failed.".format(font_dir_abs))
                   raise
               else:
                   print("New directory successfully created")

        # Create and configure ImageSurface
        figure_dimensions = Vector(self.image_size, self.image_size)
        pix = qah.ImageSurface.create(format = CAIRO.FORMAT_RGB24,
                                      dimensions = figure_dimensions)

        # Create context
        ctx = qah.Context.create(pix)
        ctx.set_source_colour(Colour.x11["white"])
        ctx.paint()
        ctx.set_source_colour(Colour.x11["black"])
        ctx.set_font_face(self.__font_face)
        ctx.set_font_size(self.font_size)

        # Center text vertically and horizontally
        self.center_text(ctx, code_point)

        # Show text and write to file
        ctx.show_text(code_point)
        pix.flush()
        pix.write_to_png(os.path.join(font_dir_abs, str(ord(code_point)) + ".png"))
        #pix.write_to_png('test.png')

    def position_text(self, context, text, x, y):
        """Put text in the intended position (relative to the origin)."
        
        Args:
            context: Qahirah context for the selected font and text size
            text: Unicode codepint for the text (character intended to be drawn)
            pos_x: X coordinate for the intended position
            pos_y: Y coordinate for the intended position
  
        Returns:
            None
        """
        # Calculate how much to move on x and y axis
        text_extents = context.text_extents(text)
        text_height = text_extents.height
        text_width = text_extents.width
        delta_x = pos_x - text_extents.x_bearing
        delta_y = pos_y - text_extents.y_bearing

        # Move in context
        context.move_to((delta_x, delta_y))

    def center_text(self, context, text):
        """Put text in the center of the image (vertically and horizontally).
    
        Args:
            context: Qahirah context for the selected font and text size
            text: Unicode codepint for the text (character intended to be drawn)
  
        Returns:
            None
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
    parser = argparse.ArgumentParser(description='Usage: \n', \
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--font_size', type=int, default=36, required=False, \
                        help='Font size for all the character')
    parser.add_argument('--image_size', type=int, default=40, required=False, \
                        help='Height and width (in number of pixels) of the image')
    parser.add_argument('--fonts', type=str, default="Noto Sans CJK SC", required=False, \
                        nargs='+', help="")
    args = parser.parse_args()
    
    # Parse arguments
    font_size = args.font_size
    image_size = args.image_size
    margin = (image_size - font_size) / 2
    font = "Noto Sans CJK SC"
    text = "\u6642" # Han character: 時
    #text = "\u0061" # Latin character: a

    # Raise exception for invalid input
    if font_size > image_size:
        raise ValueError('Argument image_size must be larger than font_size.')

    # Find font face and set size
    ft = qah.get_ft_lib()
    ft_face = ft.find_face(font)
    if ft_face.family_name != font:
        raise ValueError("Specified font {} cannot be found.".format(font))

    # Figure configuration
    cairo_face = qah.FontFace.create_for_ft_face(ft_face)
    figure_dimensions = Vector(image_size, image_size)

    # Create image
    pix = qah.ImageSurface.create(format = CAIRO.FORMAT_RGB24,
                                  dimensions = figure_dimensions)

    # Create context
    ctx = \
            (qah.Context.create(pix)
                .set_source_colour(Colour.x11["white"])
                .paint()
                .set_source_colour(Colour.x11["black"])
                .set_font_face(cairo_face)
                .set_font_size(font_size)
            )
    #import pdb;pdb.set_trace()
    #center_text(ctx, text, image_size)
    ctx.show_text(text)
    pix.flush()
    pix.write_to_png("test.png")
    
