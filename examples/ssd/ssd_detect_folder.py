#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree
print 
import datetime

now = datetime.datetime.now()

# Define all the folders
caffe_root = '/root/caffe'

input_dir = "/root/input"
output_dir = "/root/output"

output_unique = os.path.join(output_dir, now.strftime("%Y-%m-%d_%H:%M"))
boxed_images_subsubdir = 'with_boxes'
xml_subsubdir = 'xml'


xml_dir = os.path.join(output_unique,xml_subsubdir)

boxed_dir =  os.path.join(output_unique, boxed_images_subsubdir)


for directory in [xml_dir, boxed_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get list of image files
included_extensions = ['jpg', 'png', 'jpeg']
images = [fn for fn in os.listdir(input_dir)
              if any(fn.endswith(ext) for ext in included_extensions) 
              and fn[0:2] != "._"]

os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    
    for image in images:
        input_image_name = os.path.join(input_dir, image)
        output_image_name = os.path.join(boxed_dir, image)
        xml_name = os.path.join(xml_dir, os.path.splitext(image)[0]) + ".xml"
        # xml_file = open(xml_name, "w")
        print(input_image_name)
        result = detection.detect(input_image_name)
        print result

        img = Image.open(input_image_name)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        width, height = img.size
        print width, height
        #with open(xml_name, "w") as xml_file:

        def generate_xml_stem(image_name = image, image_width = width, image_height = height):
            root=etree.Element("annotation")

            folder = etree.SubElement(root, "folder")
            folder.text = output_dir
            filename=etree.SubElement(root,"filename")
            filename.text=image_name

            path = etree.SubElement(root,"path")
            path.text = "."
            source = etree.SubElement(root,"source")
            database = etree.SubElement(source,"database")
            database.text = "Unknown"

            size = etree.SubElement(root, "size")
            width = etree.SubElement(size, "width")
            height = etree.SubElement(size, "height")
            depth = etree.SubElement(size, "depth")
            width.text = str(image_width)
            height.text = str(image_height)
            depth.text = str(3)

            segmented = etree.SubElement(root, "segmented")
            segmented.text = str(0)



            return root

        def add_detected_object_to_xml(xml_root, coords, category):
            obj = etree.SubElement(xml_root, "object")
            name = etree.SubElement(obj, "name")
            name.text = category
            pose = etree.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = etree.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = etree.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = etree.SubElement(obj, "bndbox")
            xmin = etree.SubElement(bndbox, "xmin")
            ymin = etree.SubElement(bndbox, "ymin")
            xmax = etree.SubElement(bndbox, "xmax")
            ymax = etree.SubElement(bndbox, "ymax")
            xmin.text = coords["xmin"]
            ymin.text = coords["ymin"]
            xmax.text = coords["xmax"]
            ymax.text = coords["ymax"]
            return xml_root

        xml_root = generate_xml_stem(image)




        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            coords = {  "xmin": str(xmin),
                        "ymin": str(ymin), 
                        "xmax": str(xmax), 
                        "ymax": str(ymax)}
            category = str(item[-1])
            xml_root = add_detected_object_to_xml(xml_root, coords, category)

            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            draw.text([xmin, ymin], str(round(int(item[-2], 2))) + " " + category, (0, 0, 255))
            
        tree = etree.ElementTree(xml_root)
        tree.write(xml_name, pretty_print=True, xml_declaration=True)
        img.save(output_image_name)
        # xml_file.close()


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/root/caffe/data/VOC0712/labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='/root/ssd/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='/root/ssd/'
                        'weights.caffemodel')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
