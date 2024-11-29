import numpy as np
import gradio as gr
import tensorflow as tf

def get_prediction(input_img):
    print(input_img.shape)
    input_img = tf.image.resize(input_img, [224, 224])
    print(input_img.shape)
    input_img = tf.reshape(input_img, (1, 224, 224, 3))
    print(input_img.shape)
    input_img /= 255.0
    pred = model.predict(input_img)
    print(pred)
    results = {
        'Deciduous' : pred[0][0],
        'Permanent' : pred[0][1]
    }
    return results

demo = gr.Interface(get_prediction, inputs="image", outputs= "label", title= "Deciduous/Permanent Tooth Classification")
if __name__ == "__main__":
    model = tf.keras.models.load_model('DenseNet201.keras', compile= False)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    demo.launch(share= False)

