import os, sys
import cv2


def convert_to_binary(img_grayscale):
    # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    img_binary = cv2.adaptiveThreshold(img_grayscale,
                                       maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv2.THRESH_BINARY,
                                       blockSize=99,
                                       C=8)
    return img_binary

if __name__ == "__main__":
    assert len(sys.argv) == 2, '[USAGE] $ python holding.py image_5.jpg'
    input_image_path = sys.argv[1]

    assert os.path.isfile(input_image_path), 'Image not found @ %s' % input_image_path
    # read color image with grayscale flag: "cv2.IMREAD_GRAYSCALE"mage_path

    img_grayscale = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)  # shape: (960, 960)
    # print grayscale image
    cv2.imwrite('grey_%s' % input_image_path, img_grayscale)
    print('Saved grayscale image @ grey_%s' % input_image_path)

    img_binary = convert_to_binary(img_grayscale)
    cv2.imwrite('binary_%s' % input_image_path, img_binary)
    print('Saved binary image @ binary_%s' % input_image_path)