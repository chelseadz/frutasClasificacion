import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tensorflow_hub import KerasLayer
from collections import deque
from tensorflow.keras.models import load_model


model = tf.keras.models.load_model(
       ("C:/Users/te533640/Documents/personal/6sem/RN/proyecto/modelfruit.h5"),
       compile = False
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

labels =['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']

def process_frame(frame):
    # Redimensionar el frame al tamaño esperado por el modelo
    resized_frame = cv2.resize(frame, (100, 100))

    # Normalizar los valores de píxel en el rango [0, 1]
    normalized_frame = resized_frame / 255.0

    # Agregar una dimensión adicional al tensor del frame para que coincida con la forma esperada por el modelo
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Realizar la predicción utilizando el modelo
    predictions = model.predict(input_frame)

    # Obtener la clase con la mayor probabilidad
    predicted_class = np.argmax(predictions)

    # Obtener la etiqueta correspondiente a la clase predicha
    predicted_label = labels[predicted_class]

    # Imprimir la etiqueta en la consola
    return predicted_label

st.title("Clasificación de frutas")

frame_placeholder = st.empty()

#Put the start and stop buttons in the same row
col1_start, col2_stop = st.columns([4,1])

with col1_start:
    start_button = st.button("Start")
with col2_stop:
    stop_button = st.button("Stop")

prediction = st.empty()

if start_button:
        cap = cv2.VideoCapture(0) 
        
        while cap.isOpened() and not stop_button:
            # Get the frame
            ok, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ok:
                print("Error while reading camera frame")
                break

            frame_placeholder.image(frame_rgb, channels="RGB")
            flower = process_frame(frame_rgb)
            prediction.header(flower)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
            
