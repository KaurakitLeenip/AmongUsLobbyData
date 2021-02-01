import cv2
from skimage.metrics import structural_similarity as ssim
import pytesseract
import os
import numpy as np
import glob
from collections import OrderedDict
import Levenshtein as lv
import pprint

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_RGB2GRAY)
IMAGE_DEAD = cv2.cvtColor(cv2.imread("image3.png"), cv2.COLOR_RGB2GRAY)

"""COORDS of the names in the lobby screen"""
COORDS = [
    {'x1': 400, 'x2': 700, 'y1': 215, 'y2': 275},
    {'x1': 1050, 'x2': 1350, 'y1': 215, 'y2': 275},
    {'x1': 400, 'x2': 700, 'y1': 350, 'y2': 410},
    {'x1': 1050, 'x2': 1350, 'y1': 350, 'y2': 410},
    {'x1': 400, 'x2': 700, 'y1': 490, 'y2': 550},
    {'x1': 1050, 'x2': 1350, 'y1': 490, 'y2': 550},
    {'x1': 400, 'x2': 700, 'y1': 630, 'y2': 690},
    {'x1': 1050, 'x2': 1350, 'y1': 630, 'y2': 690},
    {'x1': 400, 'x2': 700, 'y1': 760, 'y2': 820},
    {'x1': 1050, 'x2': 1350, 'y1': 760, 'y2': 820}
]
RES = {}

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=5.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def preprocess_image(image):
    """removes noise and threshes an image"""
    # image = unsharp_mask(image)
    image = cv2.medianBlur(image, 3)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return image


def get_ssims(image, img_comparison):
    """returns the structural similarity score between an image and the base image"""
    return ssim(img_comparison, image, multichannel=True)


def get_images_from_video(path):
    """takes a path to a video to record a screenshot every 90 seconds
    checks if the image is enough like a meeting screen and preprocesses and saves it
    """
    cap = cv2.VideoCapture(path)
    ms_interval = 120000
    ms_count = 1
    success = True

    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_interval*ms_count)
        success, image = cap.read()

        if not success:
            # end of video
            break

        ms_count += 1
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if get_ssims(image, IMAGE_BASE) > 0.8 or get_ssims(image, IMAGE_DEAD) > 0.8:
            #if image is similar to a meeting screen, split it up into players
            print("found meeting at time {0}s".format((ms_interval*ms_count)/1000))
            image = preprocess_image(image)
            process_lobby(image)

    cap.release()
    cv2.destroyAllWindows()


def process_lobby(image):
    """Takes a meeting image and splits it into the names of the people in the lobby
    and then processes the image to extract text from it"""
    a = 0
    for i in COORDS:
        temp = image[i['y1']:i['y2'], i['x1']:i['x2']]
        temp = floodfill_invert_training(temp)
        a += 1
        text = pytesseract.image_to_string(temp, config='--psm 8 -l mungus').strip()
        if text in RES:
            RES[text] += 1
        else:
            RES[text] = 1


def floodfill_invert_training(img):
    """fills the background in black and inverts whole picture in grayscale
    this will use all four corners of the image as anchors for FloodFill"""
    height, width = img.shape
    mask1 = np.zeros((height + 2, width + 2), np.uint8)
    img = cv2.floodFill(img, None, (0, 0), (0, 0, 0))[1]
    img = cv2.floodFill(img, None, (0, 59), (0, 0, 0))[1]
    img = cv2.floodFill(img, None, (299, 0), (0, 0, 0))[1]
    img = cv2.floodFill(img, None, (299, 59), (0, 0, 0))[1]
    img = cv2.bitwise_not(img)
    return img


def main():
    get_images_from_video("F:/Mungus Rips/20201027.mp4")

    temp = []
    res = {}

    """Collect the results and group them by levenstein similarity"""
    for name in list(RES.keys()):
        for i in temp:
            if all(lv.ratio(name, j) > 0.65 for j in i):
                i.append(name)
                break
        else:
            temp.append([name, ])

    for grp in temp:
        res[tuple(grp)] = 0
        for name in grp:
            res[tuple(grp)] += RES[name]

    pprint.pprint(sorted(res.items(), key=lambda item: item[1], reverse=True))


def alt():
    cap = cv2.VideoCapture("F:/Mungus Rips/20201014.mkv")

    cap.set(cv2.CAP_PROP_POS_MSEC, 270000)
    success, image = cap.read()
    cv2.imwrite("image3.png", image)

    cap.release()
    cv2.destroyAllWindows()


def tesseract_test():
    images = [cv2.imread(names) for names in glob.glob(ROOT_DIR + "/images/training/*.png")]
    for image in images:
        text = pytesseract.image_to_string(image, config='--psm 8 --oem 3 -l mungus').strip()
        if text in RES:
            RES[text] += 1
        else:
            RES[text] = 1
    print(OrderedDict(sorted(RES.items(), key=lambda t: t[1])))


def check_images(image):
    """check lobby name pictures with ssim, save it if it doesnt exist"""
    i = 0
    ssims = []
    crop = cv2.imread(ROOT_DIR + "/images/cropped{0}.png".format(i))
    while crop is not None:
        i += 1
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        ssims.append(get_ssims(image, crop))
        crop = cv2.imread(ROOT_DIR + "/images/cropped{0}.png".format(i))

    if not any(x > 0.8 for x in ssims):
        cv2.imwrite(ROOT_DIR + "/images/cropped{0}.png".format(i), image)

if __name__ == "__main__":
    main()


