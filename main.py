import cv2
from skimage.metrics import structural_similarity as ssim
import pytesseract
import os
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_RGB2GRAY)
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
flag = False

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
    """sharpens, removes noise and threshes an image"""
    image = unsharp_mask(image)
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
    ms_interval = 90000
    ms_count = 1
    img_count = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_interval*ms_count)
        success, image = cap.read()

        if not success:
            # end of video
            break

        ms_count += 1
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if get_ssims(image, IMAGE_BASE) > 0.8:
            #if image is similar to a meeting screen, split it up into players
            image = preprocess_image(image)
            process_lobby(image)
            img_count += 1

    cap.release()
    cv2.destroyAllWindows()


def process_lobby(image):
    """Takes a meeting image and splits it into the names of the people in the lobby"""
    for i in COORDS:
        temp = image[i['y1']:i['y2'], i['x1']:i['x2']]
        text = pytesseract.image_to_string(temp).strip()
        if text in RES:
            RES[text] += 1
        else:
            RES[text] = 1
        check_images(temp)


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


def tesseract_test():
    for i in range(60):
        print(i)
        img1 = cv2.imread(ROOT_DIR + '/images/training/{0}.png'.format(i))
        text = pytesseract.image_to_string(img1, config='--psm 8').strip()
        if text in RES:
            RES[text] += 1
        else:
            RES[text] = 1
    print(RES)

def floodfill_invert_training(index):
    """fills the background in black and inverts whole picture in grayscale"""
    img = cv2.imread(ROOT_DIR + '/images/cropped{0}.png'.format(index))
    height, width = img.shape[:-1]
    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    img = cv2.floodFill(img, mask1, (0, 0), (0, 0, 0))[1]
    img = cv2.bitwise_not(img)
    cv2.imwrite(ROOT_DIR + '/images/training/{0}.png'.format(index), img)


def main():
    get_images_from_video("F:/Mungus Rips/20201016_772783968_Among Us.mp4")
    print(RES)
    # for i in range(71):
    #     print(i)
    #     floodfill_invert_training(i)
    # tesseract_test()

    # img1 = cv2.imread(ROOT_DIR + '/images/training/{0}.png'.format(1))
    # text = pytesseract.image_to_string(img1, config='--psm 8')
    # print(text.strip())

if __name__ == "__main__":
    main()
