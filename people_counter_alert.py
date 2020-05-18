
import numpy as np
import tensorflow as tf
import cv2
import time

class DetectorAPI:
    def __init__(this, path_to_ckpt):
        this.path_to_ckpt = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
        this.detection_graph = tf.Graph()
        with this.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(this.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        this.default_graph = this.detection_graph.as_default()
        this.sess = tf.Session(graph=this.detection_graph)

        # Definite input and output Tensors for detection_graph
        this.image_tensor = this.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        this.detection_boxes = this.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        this.detection_scores = this.detection_graph.get_tensor_by_name('detection_scores:0')
        this.detection_classes = this.detection_graph.get_tensor_by_name('detection_classes:0')
        this.num_detections = this.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Frame per second(fps):", 1/(end_time-start_time))

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(this):
        this.sess.close()
        this.default_graph.close()


if __name__ == "__main__":

    odapi = DetectorAPI(path_to_ckpt='ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb')
    threshold = 0.5
    cap = cv2.VideoCapture(0)


    while True:
        #input from usb camera
        r, video = cap.read(0)
        video = cv2.resize(img, (960, 544))
        boxes, scores, classes, num = odapi.processFrame(video)
        final_score = np.squeeze(scores)    
        count = 0
            
        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if scores is None or final_score[i] > threshold:
                count = count + 1
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #add text to sreen
        cv2.putText(video,"Person detected {}".format(str(count)),(45,70), font, 1,(255,0,0),1,cv2.LINE_AA)
        if count >= 3:
            cv2.putText(img,"Detected 3 or more than 3 people!",(100,350), font, 1,(0,0,255),1,cv2.LINE_AA)
            
        cv2.imshow("people counter", video)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
