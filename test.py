import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import argparse
import collections
from glob import glob

import cv2
import numpy as np
from numba import jit
from tqdm import tqdm
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

#for multiple thread computing
strategy = tf.distribute.MultiWorkerMirroredStrategy()


parser = argparse.ArgumentParser(description='description')
parser.add_argument('--img_path', action="store", type=str, help='path to the image folder')
parser.add_argument('--save_path', action="store", type=str, help='path to the result folder')
args = parser.parse_args()


#model initialization
start_time = time.time()
model = tf.saved_model.load('trained_model')
print('\nmodel initialized in {} seconds \n'.format(time.time() - start_time))


STANDARD_COLORS = ['SaddleBrown', 'AliceBlue', 'LawnGreen', 'OrangeRed', 'Yellow']
CLASSES = ["background", 'wire', 'small garments', 'door', 'furniture']

@jit
def draw_bounding_box_on_image(image, boxes, classes, scores):
    '''
    Draws bounding box around image.

    parameters - 

        image - input image where the boxes will be drawn
        boxes - detected bounding boxes
        classes - classes of the respective bounding box
        Scores - confidence score for every bounding box. 

    return - 

        image where the bounding boxes are drawn

    '''
    font = ImageFont.load_default()
    box_to_color_map = collections.defaultdict(str)
    box_to_display_str_map = collections.defaultdict(list)
    for i in range(len(boxes)):
        box = tuple(boxes[i])
        box_to_color_map[box] = STANDARD_COLORS[classes[i]]
        display_str = str(CLASSES[classes[i]])
        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
        box_to_display_str_map[box].append(display_str)

    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        draw = ImageDraw.Draw(image_pil); thickness=6; im_width, im_height = image_pil.size
        display_str_list=box_to_display_str_map[box]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),(left, top)], width=thickness, fill=color)

        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height


        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],fill=color)
            draw.text( (left + margin, text_bottom - text_height - margin),display_str,fill='black',font=font)
            text_bottom -= text_height - 2 * margin
        np.copyto(image, np.array(image_pil))
    return image

def detection (image):
    '''
    detect bounding box from the image.

    parameters - 

        image - input image where the boxes will be detected

    return - 

        boxes - detected bounding boxes
        classes - classes of the respective bounding box
        Scores - confidence score for every bounding box. 

    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with strategy.scope():
      input_tensor = tf.convert_to_tensor(image_rgb)
      input_tensor = input_tensor[tf.newaxis, ...]
      detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #non max suppression with score threshold and iou. 
    index = tf.image.non_max_suppression(detections['detection_boxes'],
                                         detections['detection_scores'],
                                         max_output_size = 20, 
                                         iou_threshold = 0.5,
                                         score_threshold = 0.5)
    
    boxes = [detections['detection_boxes'][i] for i in index]
    classes = [detections['detection_classes'][i] for i in index]
    scores = [detections['detection_scores'][i] for i in index]

    return boxes, classes, scores

def main (img_path, save_path):
    for path in tqdm(sorted(glob(os.path.join(img_path ,'*.jpg')))):
        image = cv2.imread(path)
        boxes, classes, scores = detection (image)
        result = draw_bounding_box_on_image(image.copy(), boxes, classes, scores)
        cv2.imwrite(os.path.join(save_path, os.path.basename(path).split('.')[0]+'.jpg'), result)


if __name__ == '__main__':
    img_path = args.img_path
    save_path = args.save_path
    main(img_path, save_path)
