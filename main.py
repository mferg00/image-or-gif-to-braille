import cv2
import os
import codecs
import numpy as np
import sys

'''
braille ascii explanation:

start byte: 0x2800

braille        example      to get char value:
numbering:     (1,4,2,6):     find char byte
1 4            o o          --- (1) (2) 3 (4) 5 (6) 7
2 5            o            | convert to binary
3 6              o          -->  1 1 0 1 0 1 0
7 8                         | reverse bit order
                            -->  0 1 0 1 0 1 1 = 43 = 2B
                            | add byte to braille start byte
                            --> 0x2800 + 2B = 0x282B = ⠫
'''

# ascii byte as pixel width/height
PX_W = 2
PX_H = 4
W_GAP = 1
H_GAP = 2


def nothing(x):
    pass


def get_dim(frame):
    try:
        height, width, _ = frame.shape
    except ValueError:
        height, width = frame.shape

    return height, width


def resize(frame, new_width):

    width, height = get_dim(frame)

    ratio = new_width / float(width)
    width, height = new_width, int(height * ratio)

    width = width - (width % (PX_W + W_GAP))
    height = height - (height % (PX_H + H_GAP))

    return cv2.resize(frame, (height, width))


def adjust_image(frame, thresh, invert, new_width, bw_input=False):

    kernel = np.ones((1, 1), np.uint8)
    frame = resize(frame, new_width)

    if not bw_input:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not invert:
        _, frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)
    else:
        _, frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY_INV)

    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    return frame


def find_image_params(frame, _width, _invert):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = resize(frame, 600)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    title = 'press s when done, or q to quit'
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('thresh', title, 127, 255, nothing)

    if _invert is None:
        cv2.createTrackbar('invert', title, 0, 1, nothing)
    else:
        invert = _invert

    if _width is None:
        cv2.createTrackbar('char width', title, 60, 500, nothing)
    else:
        width = char_to_px_width(_width)

    while True:

        thresh = cv2.getTrackbarPos('thresh', title)

        if _invert is None:
            invert = cv2.getTrackbarPos('invert', title)

        if _width is None:
            width = cv2.getTrackbarPos('char width', title)
            width = 10 if width < 10 else width

        if not invert:
            _, mask = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY_INV)

        prev = adjust_image(frame, thresh, invert, char_to_px_width(width), bw_input=True)

        cv2.imshow('preview', prev)
        cv2.imshow(title, mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.destroyAllWindows()
            break

        if key == ord('q'):
            raise SystemExit

    return thresh, invert, char_to_px_width(width)

'''
input: thresholded image frame
process: capture 2x4 pixel group, and then move to the next group
         pixel group     put into 8 wide bit
         example:        array in reverse braille order:
         255 0           1 1 1 0 1 0 0 1
         255 255
         255 0
         0   255

         every third column and 5th/6th row is skipped due to the spacing between ascii characters
output: string of braille image to be written
'''
def frame_to_braille(im):

    print(".", end='')
    height, width = get_dim(im)

    pixel_group = np.zeros(8, dtype=np.uint8)
    output = ""
    for outer_j in range(0, height - PX_H, PX_H):
        for outer_i in range(0, width - PX_W, PX_W):
            for inner_i in range(0, PX_W):
                for inner_j in range(0, PX_H):
                    index = braille_pos(inner_i, inner_j)
                    pixel_group[index] = im[outer_j + inner_j, outer_i + inner_i] / 255

            output += get_braille(pixel_group)
            pixel_group.fill(0)

        output += '\n'

    return output

'''
input: 8 wide bit array of reverse braille order
process: pack bits to one integer in reverse order, add to braille start char
output: braille char
'''
def get_braille(array):
    return chr(0x2800 + np.packbits(array, bitorder='little'))


def braille_pos(i, j):

    if j < 3:
        return j + (3 * i)
    if j == 3:
        return 6 + i
    return 0


def px_to_char_width(width):
    return None if width is None else int(np.round(width / (PX_W + W_GAP)))


def char_to_px_width(width):
    return None if width is None else int(np.round(width * (PX_W + W_GAP)))


def file_to_braille(path, output, width=None, invert=None, take=None, skip=None):

    if path.endswith(".gif"):

        print(f"converting gif: '{path}'", end=' ')
        im = cv2.VideoCapture(path)
        ret, frame = im.read()

        if not ret:
            print(" aborted, bad file")
            return

        thresh, invert, new_width = find_image_params(frame, width, invert)

        write_iter = 0
        total_iter = 1
        while take is None or write_iter < take:

            if skip is None or total_iter % skip == 0:
                frame = adjust_image(frame, thresh, invert, new_width)
                output.write(frame_to_braille(frame))
                write_iter += 1

            total_iter += 1

            ret, frame = im.read()

            if not ret:
                break

    else:

        print(f"converting file: '{path}'", end=' ')
        frame = cv2.imread(path)

        if frame is None:
            print(" aborted, bad file")
            return

        thresh, invert, new_width = find_image_params(frame, width, invert)
        frame = adjust_image(frame, thresh, invert, new_width)

        output.write(frame_to_braille(frame))

    print(" done")


if __name__ == '__main__':

    import argparse

    try:
        file = sys.argv[1]
    except IndexError:
        print("invalid command, use path to folder or filename")
        raise SystemExit

    path = '.' + file

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, help="set max image width (in char size)", required=False)
    parser.add_argument("-i", "--invert", type=int, help="invert output", required=False)
    parser.add_argument("-t", "--take", type=int, help="max frames to take from gif", required=False)
    parser.add_argument("-s", "--skip", type=int, help="frames to skip when reading a gif", required=False)

    args, _ = parser.parse_known_args()

    with codecs.open("braille.txt", "w", "utf-8-sig") as out:
        if os.path.isfile(path):
            file_to_braille(path, out, width=args.width, invert=args.invert,
                            take=args.take, skip=args.skip)

        else:
            for filename in os.listdir(path):
                file_to_braille(path + filename, out, width=args.width,
                        invert=args.invert, take=args.take, skip=args.skip)
