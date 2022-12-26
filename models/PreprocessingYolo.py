import os
from os import listdir
from os.path import splitext
import pybboxes as pbx
import cv2

# All paths to be updated later
target_coordinates_directory = './TestGroundTruth'
target_image_directory = './TestImages'

os.mkdir('./preprocessedTest')
os.mkdir('./LabelsTest')


for img, label in zip(listdir(target_image_directory), listdir(target_coordinates_directory)):
    # 1. read, resize image & save it
    image = cv2.imread(target_image_directory+ '/' + img)
    X_scale = 0.64
    Y_scale = 0.64
    h, w, c = image.shape
    size = (int(X_scale * w), int(Y_scale * h))
    resizedImage = cv2.resize(image, size)
    imgName, _ = splitext(img)
    cv2.imwrite('./preprocessedTest/' + imgName+'.jpg', resizedImage)

    # 2. Convert bbox annotation from VOC to YOLO format & save it
    height, width = size
    filename, _ = splitext(label)
    lines = []
    with open(target_coordinates_directory+ '/'+ label, 'r') as file:
        for line in file.readlines():
            l = line.split(',')
            x1 = int(l[0])
            y1 = int(l[1])
            x2 = int(l[2])
            y2 = int(l[3].split('\n')[0])

            # convert
            voc_bbox = (int(x1 * X_scale), int(y1 * Y_scale), int(x2 * X_scale), int(y2 * Y_scale))
            W, H = width, height  # WxH of the image
            xc, yc, w, h = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W, H))
            newline = ""
            newline += '0 '
            newline += str(xc) + ' '
            newline += str(yc) + ' '
            newline += str(w) + ' '
            newline += str(h)
            newline += '\n'
            lines.append(newline)
    with open('./LabelsTest/' + filename + '.txt', 'w') as f:
        for line in lines:
            f.write(line)

