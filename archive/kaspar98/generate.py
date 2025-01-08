import os
import argparse
from time import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.draw import line_aa
from skimage.transform import resize

import numpy as np

from utils import rgb2gray, largest_square, scale_nails
from canvas import init_canvas, layout_nail_positions, options_shape_type


RESOLUTION = 500


def get_aa_line(from_pos, to_pos, str_strength, picture):

    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])

    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)

    return line, rr, cc


def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength, random_nails: int = -1):

    best_cum_improvement = -99999
    best_nail_position = None
    best_nail_idx = None
    
    if random_nails > 0:
        nail_ids = np.random.choice(range(len(nails)), size=random_nails, replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:

        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)

        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc]) ** 2

        cum_improvement =  np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cum_improvement >= best_cum_improvement:
            best_cum_improvement = cum_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cum_improvement


def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None):

    start = time()
    iter_times = []

    current_position = nails[0]
    pull_order = [0]

    i = 0
    fails = 0

    while True:
        start_iter = time()

        i += 1
        
        if i % 500 == 0:
            print(f"Iteration {i}")
        
        if i_limit == None:
            if fails >= 3:
                break
        else:
            if i > i_limit:
                break

        idx, best_nail_position, \
        best_cum_improvement = find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength)

        if best_cum_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    print(f"Avg iteration time: {np.mean(iter_times)}")

    return pull_order


def pull_order_to_array_bw(order, canvas, nails, strength):
    """
    Draw a black and white pull order on the defined resolution
    """
    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength

    return np.clip(canvas, a_min=0, a_max=1)


def pull_order_to_array_rgb(orders, canvas, nails, colors, strength):

    color_order_iterators = [iter(zip(order, order[1:])) for order in orders]
    
    for _ in range(len(orders[0]) - 1):
        # pull colors alternately
        for color_idx, iterator in enumerate(color_order_iterators):
            pull_start, pull_end = next(iterator)
            rr_aa, cc_aa, val_aa = line_aa(nails[pull_start][0], nails[pull_start][1],
                                           nails[pull_end][0], nails[pull_end][1])

            val_aa_colored = np.zeros((val_aa.shape[0], len(colors)))
            for idx in range(len(val_aa)):
                val_aa_colored[idx] = np.full(len(colors), val_aa[idx])

            canvas[rr_aa, cc_aa] += colors[color_idx] * val_aa_colored * strength

    return np.clip(canvas, a_min=0, a_max=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create String Art')
    parser.add_argument('-i', action="store", dest="input_file", default="samples/CV 2023 - v2.png")
    parser.add_argument('-o', action="store", dest="output_file", default=None)
    parser.add_argument('-d', action="store", type=int, dest="side_len", default=RESOLUTION)
    parser.add_argument('-m', action="store", type=int, dest="max_len", default=RESOLUTION)
    parser.add_argument('-t', action="store", type=str, dest="shape_type", default='rectangle', choices=options_shape_type)
    parser.add_argument('-s', action="store", type=float, dest="export_strength", default=0.1)
    parser.add_argument('-l', action="store", type=int, dest="pull_amount", default=None)
    parser.add_argument('-r', action="store", type=int, dest="random_nails", default=None)
    parser.add_argument('-n', action="store", type=int, dest="nail_step", default=4)
    parser.add_argument('-r1', action="store", type=float, dest="radius1_multiplier", default=1)
    parser.add_argument('-r2', action="store", type=float, dest="radius2_multiplier", default=1)
    parser.add_argument('--wb', action="store_true")
    parser.add_argument('--rgb', action="store_true")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = 'output/' + Path(args.input_file).name

    img = mpimg.imread(args.input_file)
        
    if np.any(img > 128):
        img = img / 255
    
    if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
        img = largest_square(img)
        img = resize(img, (args.max_len, args.max_len))

    shape = (len(img), len(img[0]))
    nails = layout_nail_positions(shape, args.shape_type, 
                                        args.nail_step, args.radius1_multiplier, args.radius2_multiplier)
    print(f"Nails amount: {len(nails)}")

    if args.rgb:
        iteration_strength = 0.1 if args.wb else -0.1

        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]

        # Naively-sequential algorithm
        # TODO: adaptive blending ???
        str_pic_r = init_canvas(shape, black=args.wb)
        pull_orders_r = create_art(nails, r, str_pic_r, iteration_strength, i_limit=args.pull_amount)

        str_pic_g = init_canvas(shape, black=args.wb)
        pull_orders_g = create_art(nails, g, str_pic_g, iteration_strength, i_limit=args.pull_amount)

        str_pic_b = init_canvas(shape, black=args.wb)
        pull_orders_b = create_art(nails, b, str_pic_b, iteration_strength, i_limit=args.pull_amount)

        max_pulls = np.max([len(pull_orders_r), len(pull_orders_g), len(pull_orders_b)])
        pull_orders_r = pull_orders_r + [pull_orders_r[-1]] * (max_pulls - len(pull_orders_r))
        pull_orders_g = pull_orders_g + [pull_orders_g[-1]] * (max_pulls - len(pull_orders_g))
        pull_orders_b = pull_orders_b + [pull_orders_b[-1]] * (max_pulls - len(pull_orders_b))

        pull_orders = [pull_orders_r, pull_orders_g, pull_orders_b]

        color_image_dim = (int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier), 3)
        print(color_image_dim)

        scaled_nails = scale_nails(color_image_dim[1] / shape[1],
                                   color_image_dim[0] / shape[0], nails)

        blank = init_canvas(color_image_dim, black=args.wb)
        result = pull_order_to_array_rgb(
            pull_orders,
            blank,
            scaled_nails,
            (np.array((1., 0., 0.,)), np.array((0., 1., 0.,)), np.array((0., 0., 1.,))),
            args.export_strength if args.wb else -args.export_strength
        )

    else:
        orig_pic = rgb2gray(img) * 0.9
        
        image_dim = int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier)
        
        if args.wb:
            str_pic = init_canvas(shape, black=True)
            pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=args.pull_amount)
            blank = init_canvas(image_dim, black=True)
        else:
            str_pic = init_canvas(shape, black=False)
            pull_orders = create_art(nails, orig_pic, str_pic, -0.05, i_limit=args.pull_amount)
            blank = init_canvas(image_dim, black=False)

        scaled_nails = scale_nails(image_dim[1] / shape[1],
                                   image_dim[0] / shape[0], nails)

        result = pull_order_to_array_bw(
            pull_orders,
            blank,
            scaled_nails,
            args.export_strength if args.wb else -args.export_strength
        )

    # Save
    mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)

    guide_file = os.path.splitext(args.output_file)[0] + '.txt'
    with open(guide_file, 'w') as fwriter:
        if len(pull_orders) == 3 and isinstance(pull_orders[0], (list, tuple)):
            for color, orders in zip(['red','green','blue'], pull_orders):
                fwriter.write(color + '\n')
                fwriter.writelines(str(p) + '\n' for p in orders)
        else:
            fwriter.writelines(str(p) + '\n' for p in pull_orders)

    # print(f"Thread pull order by nail index:\n{'-'.join([str(idx) for idx in pull_orders])}")

