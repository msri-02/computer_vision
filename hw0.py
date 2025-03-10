import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import skimage
from skimage.color import rgb2gray
import imageio.v2 as imageio
from skimage.filters import threshold_otsu

def show_video(images,title):
    plt.title(title)
    for i,image in enumerate(images):
        if i == 0:
            obj = plt.imshow(image)
        else:
            obj.set_data(image)
        plt.pause(.01)
        plt.draw()

# Grayscale
def convertgray(folder_dir):
    framelist = [frame for frame in os.listdir(folder_dir)]
    imagelist = [imageio.imread(os.path.join("frames" + "\\" + frame)) for frame in framelist]
    plt.set_cmap('gray')
    grayedout = np.array([rgb2gray(img) for img in imagelist])
    return grayedout

def compute_avgimg(grayedout):
    avgarr = np.mean(grayedout, axis=0)
    plt.title("Background Image")
    plt.imshow(avgarr)
    plt.show()
    return avgarr

def compute_absvalue(firstframe, background_img):
    abs_diff = np.absolute(firstframe - background_img)
    threshold = threshold_otsu(abs_diff)
    binary = abs_diff > threshold
    return binary

def draw_boxes(image,fg):
    labels = skimage.measure.label(fg)
    regions = skimage.measure.regionprops(labels)
    plt.clf()
    plt.imshow(image,cmap='gray')
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-b', linewidth=2.5)
    plt.pause(.01)
    plt.draw()

    
def main():
    # To run: input the path below to frames folder, and call python hw0.py
    folder_dir = "C:\\Users\\msrik\\Documents\\Computer Vision\\HW0\\frames"
    
    grayedout = convertgray(folder_dir)
    show_video(grayedout, "Grayscale")

    background_img = compute_avgimg(grayedout)
    compute_absvalue(grayedout[0], background_img)

    thresholds = [compute_absvalue(frame, background_img) for frame in grayedout]
    show_video(thresholds, "Masks")

    [draw_boxes(image, fg) for image, fg in zip(grayedout, thresholds)]

if __name__ == "__main__":
    main() 