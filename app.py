from utils import visualization_utils as vis_util
from utils import label_map_util
from new import wire_detection
from final_bool import gen_expression
from new_reconst import reconstruct
from table import gen_truth_table
from open_in_logisim import gen_logisim
from capture_img import capture


import subprocess
from flask_cors import CORS, cross_origin
import tensorflow as tf
import urllib
from base64 import b64decode
import base64
import cv2
import numpy as np
import os
from PIL import Image 
import json
from flask import Flask, request, render_template,jsonify
app = Flask(__name__)
CORS(app, support_credentials=True)

MODEL_NAME='inf_graph3'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
category_index = {1: {'id': 1, 'name': 'and'}, 2: {'id': 2, 'name': 'not'}, 3:{'id': 3, 'name': 'or'}} #label_map_util.create_category_index_from_labelmap('C://Programs//Anaconda//Tensorflow//models//research//object_detection//training//objectdetection.pbtxt', use_display_name=True)
PATH_TO_LABELS='label.pbtxt'

dimensions = (800, 400)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def detection(inputImg,image_path):
    
    global predictions
       
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    coordinates = vis_util.return_coordinates(
                        image_np,
                        np.squeeze(output_dict['detection_boxes']),
                        np.squeeze(output_dict['detection_classes']).astype(np.int32),
                        np.squeeze(output_dict['detection_scores']),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.30)
  
    a=os.path.basename(image_path)
    #b=os.path.splitext(a)[0]
  
    #print(a,':',coordinates)
    newImage = np.copy(inputImg)
    predictions=[]
    for i in range(len(coordinates)):
          label=coordinates[i][5]
          #confidence=coordinates[i][4]
          top_x=coordinates[i][2]
          top_y=coordinates[i][0]
          btm_x=coordinates[i][3]
          btm_y=coordinates[i][1]
          
          
          newImage = cv2.rectangle(newImage, (top_x-5, top_y-5), (btm_x+5, btm_y+5), (255,0,0), 1)
          newImage = cv2.putText(newImage, label, (top_x, top_y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 0), 1, cv2.LINE_AA)
          
          predictions.append({'label':label,'topleft':{'x':top_x, 'y':top_y},'bottomright':{'x':btm_x, 'y':btm_y}})
    cv2.imwrite('newImage.jpg',newImage)
    
    return (newImage,predictions)      


@app.route('/')
@cross_origin(supports_credentials=True)
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def detectComponents():
    no_detection='false'
    #print('detect')
    data=request.data.decode('utf-8')
    dict_data=json.loads(data)
    imageSrc=dict_data['col1']['imageSrc']
    
    resp = urllib.request.urlopen(imageSrc)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite('inp.jpg',image)
    
    path='inp.jpg'
    darr=detection(image,path)
    #print(darr[1])
    oimage=darr[0]
    cv2.imwrite('t.jpg',oimage)
    predictions=darr[1]
    if(len(predictions)==0):
        no_detection='true'
    
    base_64=''
    with open("t.jpg", "rb") as img_file:
        base_64 = base64.b64encode(img_file.read())
    base_64=base_64.decode('utf-8')
    base_64=json.dumps(base_64)

    #process image then convert image into url and pass url as output value
    
    result = {
        "no_detection":no_detection,
        "predictions": predictions,
        "detected_image": base_64
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/reconstruct', methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def reconstructCircuit():
    missing_components=''
    base_64=''
    #print('reconstruct')
    pred_data=request.data
    dict_data=json.loads(pred_data)
    #print(dict_data['col1']['predictions'])
    
    predictions=dict_data['col1']['predictions']
    components = wire_detection(predictions)
    if(predictions==[] or components==[] or predictions == None or components == None):
        missing_components='true'
        #tkinter.messagebox.showwarning("warning","No components to reconstruct the circuit.")
    else:
        missing_components='false'
        
        components = reconstruct(components, dimensions)
        #print('expppppp')
    
    comp_data=request.data
    dict_data=json.loads(comp_data)
    #print(dict_data['col1']['components'])
    
    components=dict_data['col1']['components']
    
    expression = gen_expression(components)
    #print("from expression generation code ",expression)
       
    result = {
        "exp":expression
        }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/truth_table', methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def truth_table():
    exp_data=request.data
    dict_data=json.loads(exp_data)
    #print(dict_data['col1']['expression'])
    expression=dict_data['col1']['expression']

    lst=gen_truth_table(expression)
    cols=len(lst[0])
    #print(cols)
    dict=[]
    
    if(cols==3):
        for i in range(len(lst)):
            dict.append({'id':i,'a':lst[i][0],'b':lst[i][1],'X':lst[i][2]})
    
    if(cols==4):
        for i in range(len(lst)):
            dict.append({'id':i,'a':lst[i][0],'b':lst[i][1],'c':lst[i][2],'X':lst[i][3]})
    
    if(cols==5):
        for i in range(len(lst)):
            dict.append({'id':i,'a':lst[i][0],'b':lst[i][1],'c':lst[i][2],'d':lst[i][3],'X':lst[i][4]})
    
    
    #print(dict)
    #print('hereeeeeeeee')
    #print(lst)
    result = {
        "lst":dict,
        "cols":cols
        }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)
 
@app.route('/open_in_logisim', methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def open_in_logisim():
    #print('loggggisimmm')
    
    comp_data=request.data
    dict_data=json.loads(comp_data)
    #print(dict_data['col1']['components'])
    
    components=dict_data['col1']['components']
    gen_logisim(components)
    filename='q1.circ'
    subprocess.Popen(["logisim-win-2.7.1.exe", filename])
   
    result = {
        "msg":"Completed! Try with other images too."
        }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)
 
@app.route('/capture_image', methods=['GET'])
@cross_origin(supports_credentials=True)
def capture_image():
    #print('capture')
    base_64=''
    captured_img=capture()
    if captured_img is not None:
            with open("saved_img.jpg", "rb") as img_file:
                base_64 = base64.b64encode(img_file.read())
            base_64=base_64.decode('utf-8')
            base_64=json.dumps(base_64)
    
    #print(base_64)
    result = {
        "captured_img":base_64
        }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)    
    

if __name__ == '__main__':
    app.run()