import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import io
from PIL import Image
import os

# Disabling warnings.
st.set_option('deprecation.showfileUploaderEncoding', False)

# New function to be deleted.
def predict_image(predict_button, image, pil_image):
    st.write("So, The image contains a......")
    # new = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    pil_image = pil_image.resize((100, 100))
    new = np.asarray(pil_image)
    # In case of grayScale images the len(img.shape) == 2

    background = Image.new("RGB", pil_image.size, (255, 255, 255))
    background.paste(pil_image, mask=pil_image.split()[3]) # 3 is the alpha channel
    new = np.asarray(background)


    # if len(new.shape) > 2 and new.shape[2] == 4:
    #     #convert the image from RGBA2RGB
    #     new = cv2.cvtColor(new, cv2.COLOR_BGRA2BGR)
    # # print(new.shape)
    
    data_generator = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True
    )

    st.write("Wait for it....")

    image = np.expand_dims(new, 0)
    new = data_generator.flow(image)

    # loading the model
    model_path = os.path.join("support_models", "cats_dogs_new")
    model = load_model(os.path.join(os.getcwd(), model_path))

    # Making the prediction
    prediction = model.predict(new)

    st.write("Wait for it.....")
    if np.argmax(prediction) == 0:
        st.write("### Cat")
    else:
        st.write("### Dog")


def cats_dogs(sidebar_slots, predict_button):
    st.write("## Its raining cats and dogs.")
    image_path = os.path.join(os.getcwd(), "support_models")
    image_path = os.path.join(image_path, "images")
    image_path = os.path.join(image_path, "cats_and_dogs.jpg")
    banner = Image.open(image_path)
    st.image(banner, use_column_width = True)
    st.write("Upload the image that you want to predict in the sidebar and then predict")
    st.write("> **Note** : Upload only the images that contain a cat or a dog.")
    image = sidebar_slots[0].file_uploader("Upload an image", type = ["png", "jpg", "jpeg"])
    if image:
        pil = Image.open(image).convert("RGBA")
        np_image = np.asarray(pil)
        st.image(np_image)
    
    if predict_button:
        predict_image(predict_button, np_image, pil)