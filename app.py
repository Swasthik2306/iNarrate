import streamlit as st
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from tensorflow.keras.models import Model

st.set_page_config(page_title="Image Caption Generator")




st.title('Image Caption Generator')

uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])

max_length = 35
model = load_model('model25.h5')
vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# generate caption for an image

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None

def predict_caption(model, image, tokenizer, max_length):

    # add start tag for generation process
    in_text = 'startseq'

    # iterate over the max length of sequence
    for i in range(max_length):

        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text





if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = Image.open(io.BytesIO(file_bytes))

    # Resize the image to (224, 224)
    resized_image = image.resize((224, 224))

    # Convert the resized image to a NumPy array
    image_array = np.array(resized_image)

    # Reshape data for the model
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Preprocess image for VGG
    image_array = preprocess_input(image_array)

    # Extract features
    feature = vgg_model.predict(image_array, verbose=0)

    # Predict from the trained model
    predicted_value = predict_caption(model, feature, tokenizer, max_length)

    # Resize the original image to (360, 360)
    opencv_image = image.resize((360, 360))

    # Display the resized image
    st.image(opencv_image, channels="RGB")

    # Display the predicted caption
    st.header("Predicted caption is")
    st.write(" ".join(predicted_value.split()[1:-1]))

    
    # # Create two columns for image and text
    # col1, col2, col3 = st.columns([1, 2, 3])

    # # Display the image in the first column
    # with col1:
    #     st.image(opencv_image, channels="BGR", width=360)

    # # Display the predicted text in the second column
    # with col3:
    #     st.header("Predicted caption is")
    #     st.write(" ".join(predictied_value.split()[1:-1]))

