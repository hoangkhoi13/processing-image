import sys
import cv2
import numpy as np

IMG_PATH = 'f269af61c6212f7f7630.jpg'
img_haha =  'happypolla.jpg'
# read image
img = cv2.imread(IMG_PATH,cv2.IMREAD_REDUCED_COLOR_2)
print(IMG_PATH, img.shape)  # (720, 960, 3)

img_pad = np.zeros([1000, 1000, 3])
img_pad += 100      # grey color BGR: [100, 100, 100] <=> hex: #646464
img_pad[100:850, 20:520,:] = img
cv2.imwrite('girl_xinh_2_pad.jpg', img_pad)


def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha * img + beta, dtype=int)  # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new


if __name__ == "__main__":
    alpha = 0.5
    beta = 1
    if len(sys.argv) == 3:
        alpha = float(sys.argv[1])
        beta = int(sys.argv[2])
    img = cv2.imread('girl_xinh_2_pad.jpg')  # [height, width, channel]

    # change image brightness g(x,y) = alpha*f(x,y) + beta
    img_new = change_brightness(img, alpha, beta)

    cv2.imwrite('img_3_new.jpg', img_new)
    print("%d",alpha)
    print("%d",beta)