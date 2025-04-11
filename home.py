import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import collections
import webcolors
st.title('Colour Segmentation')

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
import webcolors

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.markdown('### Uploaded Image')
    st.image(image, width=min(500, image.size[0]))
    image_np = np.array(image)
    image_np = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(image_np)
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_.astype(int)  # (n_clusters, 3)
    # Count how many pixels belong to each cluster
    label_counts = collections.Counter(labels)

    # Sort centers by most common
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    sorted_colors = [colors[i] for i in sorted_labels]
    st.markdown("### Top 3 Most Frequent Colors")
    cols = st.columns(len(sorted_colors))

    for col, color in zip(cols, sorted_colors):
        rgb = tuple(color)
        hex_color = '#%02x%02x%02x' % rgb
        col.markdown(
            f"<div style='width:100px; height:100px; background-color:{hex_color}; border-radius:5px;'></div>",
            unsafe_allow_html=True
        )
        col.markdown(f"<center>{hex_color}</center>", unsafe_allow_html=True)




