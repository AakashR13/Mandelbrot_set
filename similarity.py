from PIL import ImageChops, Image
import matplotlib.pyplot as plt 
import numpy as np
mandelbrot_non = "./res/mandelbrot_nonaccel.png"
mandelbrot_acc = "./res/mandelbrot_acc.png"
mandelbrot_3_non =  "./res/triple_mandelbrot_nonaccel.png"
mandelbrot_3_acc = "./res/triple_mandelbrot_acc.png"
def img_compare(img1,img2):
    actual_error = 0
    im1 = Image.open(img1)
    x = np.array(im1.histogram())

    im2 = Image.open(img2)
    y = np.array(im2.histogram())

    try:
        if len(x) == len(y):
            error = np.sqrt(((x - y) ** 2).mean())
            error = str(error)[:2]
            actual_error = float(100) - float(error)
        diff = ImageChops.difference(im1, im2).getbbox()
        print(f"Img1:{img1}")
        print(f"Img2:{img2}")
        print('Matching Images In percentage: ', actual_error,'\t%' )
        f = plt.figure()
        text_lable = str("Matching Images Percentage" + str(actual_error)+"%")
        plt.suptitle(text_lable)
        f.add_subplot(1,2, 1)
        plt.imshow(im1)
        f.add_subplot(1,2, 2)
        plt.imshow(im2)
        plt.show(block=True)

    except ValueError as identifier:
        f = plt.figure()
        text_lable = str("Matching Images Percentage " + str(actual_error)+"%")
        plt.suptitle(text_lable)
        f.add_subplot(1,2, 1)
        plt.imshow(im1)
        f.add_subplot(1,2, 2)
        plt.imshow(im2)
        plt.show(block=True)
        print('identifier: ', identifier)

img_compare(mandelbrot_non,mandelbrot_acc)

img_compare(mandelbrot_3_non,mandelbrot_3_acc)