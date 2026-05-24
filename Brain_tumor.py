import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
from glob import glob
from keras.models import load_model
plt.style.use("ggplot")

from utils import *
from unet import *

def run(INPUT_IMAGE):
    im_width = 256
    im_height = 256

    image_filenames_train = []
    mask_files = glob('lgg-mri-segmentation/kaggle_3m/*/*_mask*')

    for i in mask_files:
        image_filenames_train.append(i.replace('_mask', ''))

    model = load_model('unet.hdf5', custom_objects={'dice_coefficients_loss': dice_coefficients_loss, 'iou': iou, 'dice_coefficients': dice_coefficients  } )

    img = INPUT_IMAGE
    img = cv2.resize(img, (im_height, im_width))
    img = img/255
    img = img[np.newaxis, :, :, : ]
    predicted_img = model.predict(img)

    intermi=np.zeros((256,256))

    output=0
    for i in range(256):
        for j in range(256):
            if np.squeeze(predicted_img)[i][j] > 0.5:
                intermi[i][j]=255
                output+=1

    if output>50:
        colored_image = np.squeeze(img )
        grayscale_image = np.array(intermi, dtype=np.uint8)
        _, mask = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        grayscale_image_colored = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        blended_image = cv2.bitwise_and(colored_image, colored_image, mask=inv_mask)
        blended_image=(blended_image*255).astype(np.uint8)
        inv_mask2 = cv2.bitwise_not(grayscale_image_colored)
        blended_image = cv2.add(blended_image, inv_mask2)
        return blended_image,"Tumor Detected!!!"
    else:
        with open('/Users/dev/programming/KrackHack/static/images/safe.jpeg', 'rb') as f:
            image_data = f.read()
        
        # Convert image data to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode the image data into a NumPy array
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image,"Tumor Not Detected"

            