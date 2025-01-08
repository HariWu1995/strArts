import math
import numpy as np

from skimage.draw import ellipse_perimeter


options_shape_type = ['circle','rectangle']


def init_canvas(shape, black: bool = False):
    if black:
        return np.zeros(shape)
    else:
        return np.ones(shape)


def create_rectangle_nail_positions(shape, step=2):
    height, width = shape

    nails_top = [(0, i) for i in range(0, width, step)]
    nails_left = [(i, 0) for i in range(1, height-1, step)]
    nails_bot = [(height-1, i) for i in range(0, width, step)]
    nails_right = [(i, width-1) for i in range(1, height-1, step)]

    nails = nails_top + nails_right + nails_bot + nails_left
    return np.array(nails)
    

def create_circle_nail_positions(shape, step=2, r1_mult=1, r2_mult=1):
    height, width = shape

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1

    rr, cc = ellipse_perimeter(centre[0], centre[1], int(radius*r1_mult), int(radius*r2_mult))

    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: math.atan2(c[0] - centre[0], c[1] - centre[1]))
    nails = nails[::step]

    return np.asarray(nails)


def layout_nail_positions(shape, shape_type: str = 'circle', 
                                        step: int = 2, r1_mult: int = 1, r2_mult: int = 1):
    
    assert shape_type in options_shape_type, \
        f"{shape_type} is not supported! Please choose either in {options_shape_type}"

    if shape_type == 'rectangle':
        return create_rectangle_nail_positions(shape, step=step)

    if shape_type == 'circle':
        return create_circle_nail_positions(shape, step=step, r1_mult=r1_mult, r2_mult=r2_mult)


