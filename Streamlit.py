import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image



def load_model():
    model = tf.keras.models.load_model('Intermediate Class AMC.keras')
    return model

def preprocessing_image(image):
    target_size = (64,64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict(model,image):
    return model.predict(image,batch_size=1)

def interpret_prediction(prediction):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        predicted_class = 0 if score <= 0.5 else 1
        confidence_score = [score, 1-score,0]
    else:
        confidence_score = prediction[0]
        predicted_class = np.argmax(confidence_score)
    return predicted_class, confidence_score

def main():
    st.set_page_config(
        page_title = 'pet image classifier',
        layout = 'centered'
        
    )

    st.title('anjing kucing')

    try:
        model = load_model()
    except Exception as err:
        st.error(f"error : {str(err)}")
        return
    
    uploader = st.file_uploader("Pilih Gambar Anjing/Kucing Dalam Forma JPG,JPEG atau PNG", type=('jpg','jpeg', 'png'))

    if uploader is not None:
        try:
            col1,col2 = st.columns([2,1])
            with col1:
                image = Image.open(uploader)
                st.image(image, caption="Ini Gambar", use_column_width=True)
            with col2:
                if st.button('Classify', use_container_width=True):
                    with st.spinner('Sedang Menghitung'):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model,processed_image)
                        predicted_class, confidence_score = interpret_prediction(prediction)
                        class_name = ['anjing', 'kucing']
                        result = class_name[predicted_class]
                        st.success(f"Hasil Prediksi : {result.capitalize()}")

                        confidence_percent = int(confidence_score[predicted_class] * 100)
                        st.write(f"Kemiripan: {confidence_percent}%")
                        progress_bar = st.progress(confidence_percent)
                        progress_bar.progress(confidence_percent)  # Update the progress bar
                        
        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("pilih file yang benar")


if __name__ == '__main__':
    main()
