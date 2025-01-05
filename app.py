import streamlit as st
from qr_detector import qrLocator, removeObscure, getQRData
from qr_decoder import rawQRParse
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import numpy as np
import cv2

# Function to read QR code using the provided modules
def readQR(image, verbose=0):
    try:
        im, hlines, vlines = qrLocator(image, verbose=verbose)
        mask = removeObscure(im)
        qrdata = getQRData(im, hlines, vlines, mask)

        if verbose >= 1:
            st.write("Detected QR Code:")
            plt.imshow(im)
            plt.axis('off')
            st.pyplot(plt)
        
        return rawQRParse(qrdata)
    except Exception as e:
        if verbose >= 1:
            st.write(f"Error: {e}")
        return "An error occurred"

# Streamlit App UI
st.title("QR Code Reader")

st.write("Upload an image containing a QR code, and this app will decode it.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process the image to decode the QR code
    st.write("Processing the image...")
    try:
        result = readQR(tmp_file_path, verbose=1)
    except Exception as e:
        st.error("Failed to detect the QR code.")
    # Display the result
    if result:
        try:
            decoded_data = result.decode('utf-8')
            st.success("Decoded QR Code Data:")
            st.write(decoded_data)
        except Exception as e:
            st.error("Failed to decode the QR code.")

    else:
        st.error("Failed to decode the QR code.")
