from os import listdir
from os.path import splitext
import pybboxes as pbx
import cv2

# Directory path to be updated later
target_directory_coordinates = './TrainGroundTruth'
target_directory_images = './TrainImages'


for img, label in zip(listdir(target_directory_images), listdir(target_directory_coordinates)):

    image = cv2.imread(target_directory_images+'/'+img)

    # get width and height
    height, width = image.shape[:2]

    # get file name of my .txt
    filename, _ = splitext(label)

    #store lines read from file in this list
    lines = []

    #open & read my txt
    with open(target_directory_coordinates+'/' + label, 'r') as file:
        for line in file.readlines():
            l = line.split(',')
            x1 = int(l[0])
            y1 = int(l[1])
            x2 = int(l[2])
            y2 = int(l[3].split('\n')[0])

            # convert
            voc_bbox = (x1, y1, x2, y2)
            W, H = width, height # WxH of the image
            xc, yc, w, h = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W, H))
            newline = ""
            newline += '0 '
            newline += str(xc) + ' '
            newline += str(yc) + ' '
            newline += str(w) + ' '
            newline += str(h)
            newline += '\n'
            lines.append(newline)
    with open(filename + '.txt', 'w') as f:
        for line in lines:
            f.write(line)



