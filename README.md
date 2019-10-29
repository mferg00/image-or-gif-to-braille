# image-or-gif-to-braille

Can convert images or gifs into braille, with custom sliders for better results


**How to add images/gifs:**
Place folder(s)/file(s) in the directory of main.py

**How to print an image/gif:**
run 'python main.py file.extension'

**How to print all images/gifs in a folder:**
run 'python main.py path_to_imgs/'

**What each image slider does:**

*Threshold:*
> find the point in the image that has the best amount of detail

*Invert:*
> invert the image

*Width:*
> set a maximum width to convert to (one character is 1 width)


**Optional arguments:**

*-w *
> set a fixed maximum width to convert to for all images (removes slider)

*-i *
> invert all images (0 = no invert, !0 = invert) (removes slider)

*-t *
> take x amount of frames from a gif file

*-s *
> skip x amount of frames in a gif file
