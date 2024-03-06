import os
import sys
sys.path.append('..')
from src import nns
from src import embedders
from src import segmodel
import torch
from demo_functions import run_image
import streamlit as st
from src import defdevice
st.set_page_config(layout="wide")

if torch.cuda.is_available():
    defdevice.force_device('cuda:0')

image_embr = embedders.M2FImageEmbedder()
text_embr = embedders.CLIPTextEmbedder()
model = nns.Linear(1, 1)

st.header('From Captions to Pixels: Open-Set Semantic Segmentation without Masks')
uploaded_file = st.file_uploader("Choose an image...")
input_labels = st.text_input("Input labels", value = 'tree, dirt road')
st.write("The chosen labels are", input_labels)
option = st.selectbox("Choose model", ('CSTableModel', 'CSRUGDFineTuned'))

if uploaded_file is not None and input_labels is not None:
    model.load(f"../weights/{option}")
    smodel = segmodel.CSModel(image_embr, text_embr, model)

    labels = []
    for label in input_labels.split(","):
        labels.append(label.strip())

    run_image(smodel, uploaded_file, labels, web = True)
