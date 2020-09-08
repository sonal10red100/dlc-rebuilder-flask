# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 00:34:02 2020

@author: HP
"""

import cv2 
def capture():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true as long as the webcam is running
            print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                img_new=cv2.resize(frame, (800,400),interpolation=cv2.INTER_AREA)
                img_new=cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                cv2.destroyAllWindows()
                return img_new
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        