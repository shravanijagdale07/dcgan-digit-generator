# DCGAN Handwritten Digit Generator

A Deep Convolutional GAN (DCGAN) trained on the MNIST dataset to generate new handwritten digits (0 to 9), deployed as a web app.

## Live Demo

**https://dcgan-digit-generator-nhnixfah6m83jxq4zwsqqj.streamlit.app/**

Open the link, choose the number of digits and a seed, then click **Generate**.

> The app runs on the free Streamlit Community Cloud and goes to sleep after about 12 hours without visitors. If you see a "wake up" button, click it and wait a minute.

## What This Project Does

A GAN has two neural networks that compete against each other:

- **Generator:** turns 100 random numbers (noise) into a 28x28 image of a digit.
- **Discriminator:** looks at an image and decides whether it is real (from MNIST) or fake (from the Generator).

During training, the Discriminator gets better at spotting fakes, which forces the Generator to produce more realistic digits. After training, only the Generator is needed to create new digits.

```
Random noise (100 values) -> Generator -> Fake 28x28 digit -+
                                                            +-> Discriminator -> Real or Fake?
Real MNIST digit -------------------------------------------+
```

## Project Journey

| Stage | Description |
|---|---|
| 1. Baseline GAN | Simple GAN with Dense layers, trained for 20,000 steps |
| 2. Bug fix | The Discriminator was frozen by `trainable = False`, so it could not learn. Fixed with a custom `tf.GradientTape` training step |
| 3. DCGAN | Upgraded to convolutional layers (Conv2D and Conv2DTranspose) for sharper digits |
| 4. Evaluation | A CNN classifier trained on real MNIST judged 1,000 generated digits |
| 5. Deployment | Streamlit web app hosted on Streamlit Community Cloud |

## Model Architecture

**Generator**

```
Input (100) -> Dense (7x7x128) -> BatchNorm -> LeakyReLU -> Reshape (7,7,128)
-> Conv2DTranspose (64, stride 2) -> BatchNorm -> LeakyReLU        # 14x14
-> Conv2DTranspose (1, stride 2, tanh)                             # 28x28
```

**Discriminator**

```
Input (28,28,1) -> Conv2D (64, stride 2) -> LeakyReLU -> Dropout (0.3)   # 14x14
-> Conv2D (128, stride 2) -> LeakyReLU -> Dropout (0.3)                  # 7x7
-> Flatten -> Dense (1, sigmoid)
```

## Training Details

| Setting | Value |
|---|---|
| Dataset | MNIST (60,000 training images), scaled to [-1, 1] |
| Framework | TensorFlow / Keras |
| Latent dimension | 100 |
| Batch size | 128 |
| Training steps | 10,000 |
| Optimizer | Adam (learning rate 0.0002, beta_1 = 0.5) for both networks |
| Loss | Binary cross-entropy |
| Hardware | Google Colab, T4 GPU |

Training health (final steps): Discriminator accuracy about 61% to 75%, Discriminator loss about 0.57 to 0.65, Generator loss about 0.65 to 1.07. This is a stable balance between the two networks.

## Evaluation Results

A small CNN classifier was trained on real MNIST and used to judge 1,000 generated digits:

| Metric | Result |
|---|---|
| Average classifier confidence | 0.873 |
| Images with confidence above 0.9 | 63.0% |
| Digit counts (0 to 9) | 84, 114, 77, 142, 121, 122, 77, 144, 38, 81 |

All ten digit classes were generated, so there was no mode collapse. Digit 8 was the least frequent and some images are still unclear.

## Repository Structure

```
.
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── dcgan_generator.keras   # Trained Generator model
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address shown in the terminal.

## Limitations and Future Work

- Trained for only 10,000 steps, so some digits are blurry or malformed. Longer training should improve quality.
- The model is unconditional: the user cannot choose which digit is generated.
- Planned: a **Conditional GAN (cGAN)** so the user can select a digit from 0 to 9, and evaluation with a standard metric such as FID.

## Tech Stack

Python, TensorFlow / Keras, NumPy, Matplotlib, Streamlit, Google Colab, GitHub.

## References

- Goodfellow et al., *Generative Adversarial Networks*, 2014.
- Radford, Metz and Chintala, *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)*, 2015.
- LeCun et al., MNIST handwritten digit database.

## Author

[Your Name]
