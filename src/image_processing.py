import tensorflow as tf

#Function that loads an image
def load_and_preprocess_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img
