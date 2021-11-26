from flask import Flask, request
import numpy as np
from keras.models import load_model
import cv2
import base64
import io
from flask_cors import CORS



app=Flask(__name__)
CORS(app)
loaded_cnn=load_model(filepath='cnn_digit_recognizer') 


# converts image data url to numpy array size(28,28,1)
def convertImgURL(imgDataURL):
    image_b64 = imgDataURL.split(",")[1]
    image_bytes = base64.b64decode(image_b64)

    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) #to read image in color mode via opencv
    image= cv2.resize(image,(28,28),interpolation= cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY,) #grayscaling
    image = cv2.bitwise_not(image) #returns inverted colors

    image=np.reshape(image,(28,28,1))

    return image

# converts blob image to numpy array size(28,28,1)
def convertImg(img):
  
    in_memory_file = io.BytesIO() #creates an in-memory bytes buffer
    img.save(in_memory_file)
    image = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8) # .getvalue() returns a bytes array
    
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image= cv2.resize(image,(28,28),interpolation= cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY,) #grayscaling
    image = cv2.bitwise_not(image) #returns inverted colors

    image=np.reshape(image,(28,28,1))

    return image
    


@app.route('/',methods=['GET'])
def hello():
    return "Convolutional Neural Network DIGIT RECOGNIZER- Tej Vardhan"


@app.route('/cnnPredict',methods=['GET','POST'])
def predict():
    if request.method=='POST' and request.is_json: # if request is POST and contains json

        json_data = request.get_json() 
        print(json_data)

        if('imgDataURL' in json_data):
            imgDataURL=request.json.get('imgDataURL')
            imgNumpyArr=convertImgURL(imgDataURL)
        elif('imgData' in request.files):
            imgData=request.files['imgData']
            imgNumpyArr=convertImg(imgData)
        else:
            return "Bad Request"
        
        pred=loaded_cnn.predict(np.array([imgNumpyArr]))[0]

        return str(np.argmax(pred))


if __name__=="__main__":
    app.run(debug=True)
   
    