# Student App Template for Machine Learning

Welcome to the **Student App Template**! This project provides a simple and intuitive Streamlit app template designed specifically for middle school students. The template allows students to deploy their own machine learning models, with a particular focus on image recognition models trained using [Teachable Machine](https://teachablemachine.withgoogle.com/).

## Overview

This template enables students to:
- **Deploy their image recognition models:** Easily integrate and run models directly on a web app.
- **Experiment with Machine Learning:** Learn the basics of deploying ML models without dealing with complex backend code.
- **Customize the interface:** Change descriptive text, image captions, and sidebar color to reflect their project specifics.

## Features

- **User-Friendly Interface:** A clean and modern UI that supports image uploads and live image capture via a camera.
- **Model Information Panel:** A sidebar that displays key project details, example images, and model design information.
- **Easy Deployment:** Built with Streamlit, this app template can be deployed on Streamlit Cloud with minimal setup.
- **Light Theme Enforcement:** The app is configured to use the light theme by default, ensuring a consistent look and feel.

## How to Use This Template

1. **Customize Text and Colors:**
   - Update the sidebar text in the code (e.g., app title, model design details, example image captions, and author names).
   - Change the sidebar's background color by modifying the CSS in the code.
   
2. **Integrate Your ML Model:**
   - Replace the placeholder model path with the actual path to your saved image recognition model.
   - Adjust the class labels in the code to match your model's output.

3. **Deploy Your App:**
   - Push your changes to GitHub.
   - Deploy your app to [Streamlit Cloud](https://share.streamlit.io/) by connecting your GitHub repository.

## Requirements

Make sure you have the following dependencies installed:

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)

To install the required packages, run:

```bash
pip install -r requirements.txt
