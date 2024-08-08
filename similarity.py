from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import os

mandelbrot_non = "./res/mandelbrot_nonaccel.png"
mandelbrot_acc = "./res/mandelbrot_acc.png"
mandelbrot_3_non =  "./res/triple_mandelbrot_nonaccel.png"
mandelbrot_3_acc = "./res/triple_mandelbrot_acc.png"
output = "./reports/"
def img_compare(img1, img2, dst):
    # Open images and convert to numpy arrays
    im1 = Image.open(img1)
    x = np.array(im1.histogram())

    im2 = Image.open(img2)
    y = np.array(im2.histogram())

    try:
        if len(x) == len(y):
            # Calculate error with more precision
            error = np.sqrt(((x - y) ** 2).mean())
            actual_error = 100 - error  # Keep full precision
            mse_value = mean_squared_error(np.array(im1),np.array(im2));
            # Extract directory and construct filename
            base_name_img1 = os.path.splitext(os.path.basename(img1))[0]
            base_name_img2 = os.path.splitext(os.path.basename(img2))[0]
            file_name = f"{base_name_img1}_{base_name_img2}_comparison.png"
            file_path = os.path.join(dst, file_name)

            # Save results to a file
            f = plt.figure()
            text_label = f"Matching Images Percentage: {actual_error:.6f}%\nMSE: {mse_value:.6f}"
            plt.suptitle(text_label)
            f.add_subplot(1, 2, 1)
            plt.imshow(im1)
            f.add_subplot(1, 2, 2)
            plt.imshow(im2)
            # print(plt.style.available)

            # plt.style.use('grayscale')
            plt.savefig(file_path)  # Save to a file
            plt.close(f)  # Close the figure to free memory
            print(f"Img1: {img1}")
            print(f"Img2: {img2}")
            print(f"Matching Images In percentage: {actual_error:.6f}%")
            print(f"Mean Squared Error: {mse_value:.6f}")
            print(f"Comparison saved to '{file_path}'")

    except ValueError as identifier:
        dir_name = os.path.dirname(img1)
        base_name_img1 = os.path.splitext(os.path.basename(img1))[0]
        base_name_img2 = os.path.splitext(os.path.basename(img2))[0]
        file_name = f"{base_name_img1}_{base_name_img2}_comparison_error.png"
        file_path = os.path.join(dir_name, file_name)

        f = plt.figure()
        text_label = f"Matching Images Percentage: {actual_error:.6f}%"
        plt.suptitle(text_label)
        f.add_subplot(1, 2, 1)
        plt.imshow(im1)
        f.add_subplot(1, 2, 2)
        plt.imshow(im2)
        plt.savefig(file_path)  # Save to a file
        plt.close(f)  # Close the figure to free memory
        
        print(f"Comparison with error saved to '{file_path}'")
        print('identifier: ', identifier)

# def img_compare(img1, img2):
#     im1 = Image.open(img1).convert('RGB')
#     im2 = Image.open(img2).convert('RGB')

#     # Convert images to numpy arrays
#     im1_np = np.array(im1)
#     im2_np = np.array(im2)

#     # Compute MSE
#     mse_value = mse(im1_np, im2_np)
    
#     # Convert MSE to percentage similarity
#     max_pixel_value = 255  # Assuming 8-bit images
#     similarity_percentage = 100 - (mse_value / (max_pixel_value**2) * 100)

#     # Display results
#     print(f"Img1: {img1}")
#     print(f"Img2: {img2}")
#     print('Matching Images In percentage: {:.2f}%'.format(similarity_percentage))
    
#     f = plt.figure()
#     text_label = f"Matching Images Percentage: {similarity_percentage:.2f}%"
#     plt.suptitle(text_label)
#     f.add_subplot(1, 2, 1)
#     plt.imshow(im1)
#     f.add_subplot(1, 2, 2)
#     plt.imshow(im2)
#     plt.show(block=True)

img_compare(mandelbrot_non,mandelbrot_acc,output)

img_compare(mandelbrot_3_non,mandelbrot_3_acc,output)