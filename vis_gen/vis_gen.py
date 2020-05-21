
r"""Visual (image) generation module for confusable detection."""

import sys
import os
import math
import argparse
from argparse import RawDescriptionHelpFormatter

import qahirah as qah
from qahirah import CAIRO, Colour, Vector


def center_text(context, text, image_size):
    """Put text in the center of the image (vertically and horizontally).

    Args:
        context: Qahirah context for the selected font and text size
        text: Unicode codepint for the text (character intended to be drawn)
        image_size: Int, height and width of the image 
  
    Returns:
        None
    """
    
    # Calculate how much to move on x and y axis
    text_extents = ctx.text_extents(text)
    text_height = text_extents.height
    text_width = text_extents.width
    pos_x = (image_size - text_width) / 2
    pos_y = (image_size - text_height) / 2
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
    parser.add_argument('--fonts', type=str, default="Noto Sans CJK SC", required=False, nargs='+', \
                        help="")
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
    ft_face.set_char_size(size = font_size, resolution=qah.base_dpi)

    # Figure configuration
    cairo_face = qah.FontFace.create_for_ft_face(ft_face)
    margin_vector = Vector(margin, margin)
    figure_dimensions = Vector(image_size, image_size)

    # Create image
    pix = qah.ImageSurface.create \
            (
                format = CAIRO.FORMAT_RGB24,
                dimensions = figure_dimensions
            )

    # Create context
    ctx = \
            (qah.Context.create(pix)
                .set_source_colour(Colour.x11["white"])
                .paint()
                .set_source_colour(Colour.x11["black"])
                .set_font_face(cairo_face)
                .set_font_size(font_size)
            )
    text_extents = ctx.text_extents(text)


    center_text(ctx, text, image_size)
    ctx.show_text(text)
    #ctx.show_glyphs(glyphs
    pix.flush()
    pix.write_to_png("test.png")
