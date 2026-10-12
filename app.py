import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIM = 100

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dcgan_generator.keras")

model = load_model()

st.title("DCGAN Handwritten Digit Generator")
st.write("A DCGAN trained on MNIST. Click Generate to create new handwritten digits.")

num_images = st.slider("Number of digits", 1, 25, 10)
seed = st.number_input("Seed (change it for new digits)", value=42, step=1)

if st.button("Generate"):
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0, 1, (num_images, LATENT_DIM))
    imgs = (model(noise, training=False).numpy() + 1) / 2

    cols = min(num_images, 5)
    rows = int(np.ceil(num_images / cols))
    fig = plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)