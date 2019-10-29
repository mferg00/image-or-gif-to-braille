import cv2
import os
import codecs
import numpy as np
import sys

PX_W = 2
PX_H = 4


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

    new_width = new_width - (new_width % PX_W)
    ratio = new_width / width
    new_height = int(np.round(height * ratio))
    new_height = new_height - (new_height % PX_H)

    return cv2.resize(frame, (new_width, new_height))


def adjust_image(frame, thresh, invert, new_width):

    frame = resize(frame, new_width)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if new_width > 100:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    if not invert:
        _, frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)
    else:
        _, frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY_INV)

    return frame


def find_image_params(frame, _width, _invert):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = resize(frame, 600)
    title = 'press s when done'
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('thresh', title, 127, 255, nothing)

    if _invert is None:
        cv2.createTrackbar('invert', title, 0, 1, nothing)
    else:
        invert = _invert

    if _width is None:
        cv2.createTrackbar('new width', title, 60, 1000, nothing)
    else:
        width = _width

    while True:
        thresh = cv2.getTrackbarPos('thresh', title)

        if _invert is None:
            invert = cv2.getTrackbarPos('invert', title)

        if _width is None:
            width = cv2.getTrackbarPos('new width', title)

        width = 30 if width < 30 else width

        if not invert:
            _, mask = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow(title, mask)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.destroyAllWindows()
            break

    return thresh, invert, width


def frame_to_braille(im):

    print("processing...")

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

    print("done")
    return output


def get_braille(array):
    return chr(0x2800 + np.packbits(array))


def braille_pos(i, j):

    if j < 3:
        return j + (3 * i)
    if j == 3:
        return 6 + i
    return 0


def file_to_braille(path, output, width=None, invert=None):
    if path.endswith(".gif"):
        im = cv2.VideoCapture(path)
        ret, frame = im.read()

        if not ret:
            print("bad file")
            return

        thresh, invert, new_width = find_image_params(frame, width, invert)

        while True:
            frame = adjust_image(frame, thresh, invert, new_width)
            output.write(frame_to_braille(frame))

            ret, frame = im.read()

            if not ret:
                break

    else:
        frame = cv2.imread(path)

        if frame is None:
            print("bad file")
            return

        thresh, invert, new_width = find_image_params(frame, width, invert)
        frame = adjust_image(frame, thresh, invert, new_width)

        output.write(frame_to_braille(frame))


if __name__ == '__main__':

    import argparse

    try:
        file = sys.argv[1]
    except IndexError:
        print("invalid command, use path to folder or filename")
        raise SystemExit

    path = '.' + file

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, help="set max image width", required=False)
    parser.add_argument("-i", "--invert", type=int, help="invert output", required=False)

    args, _ = parser.parse_known_args()
    width = args.width
    invert = args.invert

    with codecs.open("braille.txt", "w", "utf-8-sig") as out:
        if os.path.isfile(path):
            file_to_braille(path, out, width, invert)

        else:
            for filename in os.listdir(path):
                file_to_braille(path + filename, out, width, invert)
