# -*- coding: utf-8 -*-
'''
Date: 24-08-21
Author: Suraj Kumar Aavula
'''

import argparse
import io
from logging import Logger, raiseExceptions
import boto3
import os
import cv2
import numpy as np
from PIL import Image
import json
import io
import base64
import torch
from flask import Flask, render_template, request, redirect
import datetime
import time
from flask_cors import CORS

server = Flask(__name__)
cors = CORS(server)


#creating a temp memory for storing the downloaded video from s3
dir = '/APP/downloaded_video'

s3_client = boto3.client('s3')
#bucket_name='fsi-vision-ai-train-images'
#prefix='FSI-Local-Video/damaged_car.mp4'
pro_vid_loc = '/APP/processed_video/processed_video.avi'
upload_prefix = 'processed_video/processed_video.avi'

@server.route('/')
def welcome():
    result = 'Welcome to Flask'
    return result

 
def delete_video_files(dir):

    '''Deleting the temp memory for the video file created'''
    
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def image_process(img_bytes):

    '''Processing the Image, which was sent as a byte stream form UI and '''

    server.logger.info("="*10+" Image Recieved from UI"+"="*10)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image_array


def download_video_files(bucket_name, prefix):
    
    '''Downlaods the User Video form the s3 as Presigned URL'''

    # response = s3_client.list_objects(Bucket=bucket_name, Prefix=prefix)
    # name='input1.mp4'
    # path='/APP/downloaded_video/'+name
    # for file in response['Contents']:
    #     name = file['Key'].rsplit('/', 1)
    #     s3_client.download_file(bucket_name, file['Key'],'/APP/downloaded_video/abc.mp4')
    #     if os.path.exists('/APP/downloaded_video/abc.mp4'):
    #         return '/APP/downloaded_video/abc.mp4'
    #     else:
    #         raise Exception('Video File is Not Available in Specified Location')

    server.logger.info("="*10+" Dowloding Video as URL to s3 "+"="*10)
    try:
        #s3_client.upload_file(local_path,bucket_name,prefix)
        download_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 
        'Key': prefix}, ExpiresIn = 120)
        server.logger.info("Presigned URL of Uploaded Video from UI: {}".format(download_url))
        return download_url
    except Exception as e:
        print("Error Occured {}".format(e))
        

def upload_video_files(bucket_name, local_path, prefix):
    server.logger.info("="*10+" Uploading the Processed Video to s3 "+"="*10)
    try:
        s3_client.upload_file(local_path,bucket_name,prefix)
        download_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 
        'Key': prefix}, ExpiresIn = 604800)
        server.logger.info("Download Presigned URL: {}".format(download_url))
        return download_url
    except Exception as e:
        print("Error Occured {}".format(e))


def video_processing(video_stream):

    '''Processing the downloaded video form s3
    below code gives all the required information to process the video file
    to frames and then compress them.
    Here we are skipping 5 frames alternatively to compress the video file '''

    server.logger.info("="*10+" Video Recieved from UI "+"="*10)
    cap = cv2.VideoCapture(video_stream)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFPS = int(cap.get(cv2.CAP_PROP_FPS))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    while (fc < frameCount):
        if fc % 5==0:
            ret, buf[fc] = cap.read()
        fc += 1    

    compressed_buff=[]
    for i in range(0,len(buf)):
        if i % 25 == 0:
            compressed_buff.append(buf[i])
    cap.release()
    videoArray = buf
    server.logger.info(f"DURATION: {frameCount/videoFPS}")
    return videoArray, compressed_buff, frameWidth, frameHeight, videoFPS


def predict():
    start_time = time.time()
    model = torch.hub.load(
        '/APP/yolov5', 'custom', 
        path = '/APP/weights/best.pt', source='local').autoshape()  # force_reload = recache latest code
    server.logger.info('loaded the model')
    model.eval()
    _reqest = request.form
    #for _request in request.files:
    if _reqest.get('Content-Type') == 'image/jpeg':
        img_bytes = request.files['image'].read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        server.logger.info("="*10+" Image Recieved "+"="*10)
        #processed_image = image_process(img_bytes)
        results = model(image_array, size = 416)
        results.save('/APP/pred_images'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
        results.imgs # array of original images (as np array) passed to model for inference
        results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            buffered = io.BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save(buffered, format="JPEG")
            byte_stream = {'base64': base64.b64encode(buffered.getvalue()).decode('utf-8')}
        server.logger.info("Image Result Output: {}".format(results.pandas().xyxy[0].to_json(orient="records")))
        output = json.loads(json.dumps(results.pandas().xyxy[0].to_dict('list')))
        result = {key: output[key] for key in output.keys() & {'confidence', 'name'}}
        server.logger.info('Model Predicted: {}'.format(output))
        result['data'] = 'image/jpeg'
        result.update(byte_stream)
        #server.logger.info('Returned Value: {}'.format(result))
        server.logger.info("--- Time Taken to Process a Image Request: %s seconds ---" % (time.time() - start_time))
        return result
    elif _reqest.get('Content-Type') == 'video/mp4':
        bucket_name = _reqest.get('bucket_name')
        file_path = _reqest.get('file_path')
        server.logger.info('BucketName: {}, FilePath: {}'.format(bucket_name, file_path))
        video_file = download_video_files(bucket_name, file_path)
        processed_video, compressed_video, frameWidth, frameHeight, videoFPS = video_processing(video_file)
        pred_results = []
        model.eval()
        frame_count = 1
        for frame in compressed_video:
            results = model(frame, size = 640)
            pred_results.append(results)
            #file_path = upload_prefix.format(datetime.datetime.now().strftime("%Y-%m-%d"))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        size = (frameWidth, frameHeight)
        video  = cv2.VideoWriter(pro_vid_loc, fourcc, 20.0, size, True)
        image_array = []
        for image in pred_results:
            image.imgs # array of original images (as np array) passed to model for inference
            image.render()  # updates results.imgs with boxes and labels
            for img in image.imgs:
                image_array.append(np.array(img))
                frame_count+=1
        server.logger.info('Frame Count:{}'.format(frame_count))
        frame_write = 1
        for frame in image_array:
            video.write(frame)
            frame_write+=1
        video.release()
        server.logger.info('Frames Written: {}'.format(frame_write))
        if os.path.exists('/APP/processed_video/processed_video.avi'):
            server.logger.info('Processed Video is available in given Path')
        else:
            raise Exception('Processed Video File is Not Available in Specified Location')
        drive, upload_path = os.path.splitdrive(file_path)
        path, file = os.path.split(upload_path)
        file_upload = path+ '/' + upload_prefix
        server.logger.info('Processed File Uploaded to:{}'.format(bucket_name+'/'+file_upload))
        download_url = upload_video_files(bucket_name, pro_vid_loc, file_upload)
        server.logger.info("--- Time Taken to Process a Video Request: %s seconds ---" % (time.time() - start_time))
        return download_url



@server.route("/predict", methods=["GET", "POST"])
def entry():
    if request.method == "GET":
        server.logger.info("The Model is Up and Running, got the GET Request")
        raise Exception("Invalid Request, please Send the Proper Request")    
    else:
        return predict()
