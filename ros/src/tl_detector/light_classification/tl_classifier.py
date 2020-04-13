#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
import rospy
import urllib
import tensorflow as tf
from styx_msgs.msg import TrafficLight

# cf styx_msgs/msg/TrafficLight.msg
LIGHTS = ['Red', 'Yellow', 'Green', 'Unknown', 'Unknown']
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)]
# Model output classes
MODEL_URL = 'https://github.com/frgfm/sdcnd-capstone/releases/download/v0.1.0/faster_rcnn_resnet50_coco_finetuned.pb'
MODEL_PATH = 'faster_rcnn_resnet50_coco_finetuned.pb'
CLASSES = [TrafficLight.GREEN, TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.UNKNOWN]
SAVE_RESULT = False


class TLClassifier(object):
    def __init__(self):
        self.current_light = TrafficLight.UNKNOWN

        cwd = os.path.dirname(os.path.realpath(__file__))

        model_path = os.path.join(cwd, MODEL_PATH)
        self.detection_graph = None

        # Try to download the model
        if not os.path.exists(model_path):
            try:
                urllib.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                rospy.logwarn("Unable to download model from: {}".format(MODEL_URL))
        if os.path.exists(model_path):

            # load frozen tensorflow model
            self.detection_graph = tf.Graph()
            self._load_model(model_path)

            # create tensorflow session for detection
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # end
            self.sess = tf.Session(graph=self.detection_graph, config=config)

            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.img_cnt = 0
            rospy.loginfo("TensorFlow model loaded: {}".format(model_path))
        else:
            rospy.logwarn("Unable to access model at {}".format(model_path))

    def _load_model(self, model_path):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    @staticmethod
    def render(image, rel_boxes, classes, scores):

        # Convert relatives coordinates to image coordinates
        boxes = rel_boxes
        boxes[:, [0, 2]] *= image.shape[0]
        boxes[:, [1, 3]] *= image.shape[1]

        for box, class_idx, score in zip(boxes, classes, scores):
            top, left, bot, right = box
            # Color based on detection result
            color = COLORS[CLASSES[int(class_idx) - 1]]
            cv2.rectangle(image, (left, top), (right, bot), color, 3)
            label = "{}: {:.2%}".format(LIGHTS[CLASSES[int(class_idx) - 1]], score)
            cv2.putText(image, label, (left, int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        return image

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.detection_graph is not None:
            image_np = np.expand_dims(image[..., [2, 1, 0]], axis=0)

            # Actual detection.
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores,
                     self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            min_score_thresh = .5
            is_kept = [idx for idx, score in enumerate(scores) if score >= min_score_thresh]
            boxes, scores, classes = boxes[is_kept, ...], scores[is_kept, ...], classes[is_kept, ...]
            self.current_light = CLASSES[-1]
            if len(scores) > 0:
                # Output class index are indexed from 1
                self.current_light = CLASSES[int(classes[np.argmax(scores)] - 1)]

            # Write image to disk
            if SAVE_RESULT:
                self.img_cnt += 1
                folder = '/home/workspace/CarND-Capstone/imgs/'
                processed_img = self.render(image, boxes, classes, scores)
                cv2.imwrite(folder + 'processed_img_{}.jpg'.format(self.img_cnt), processed_img)

        return self.current_light
