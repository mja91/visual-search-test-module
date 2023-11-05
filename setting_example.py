import os
import chromadb
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from glob import glob
from tqdm import tqdm

# img = Image.open("test/Bread/0.jpg")

feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16').to("cuda")

# print("Models loaded!")

# img_tensor = feature_extractor(images=img, return_tensors="pt").to("cuda")
# outputs = model(**img_tensor)

# embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()

chroma_client = chromadb.Client()

collection = chroma_client.create_collection("food")

img_list = sorted(glob("test/*/*.jpg"))

len(img_list)

embeddings = []
metadatas = []
ids = []

for i, img_path in enumerate(tqdm(img_list)) : 
    img = Image.open(img_path)
    cls = os.path.split(os.path.dirname(img_path))[-1]

    img_tensor = feature_extractor(images=img, return_tensors="pt").to("cuda")
    outputs = model(**img_tensor)

    embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()

    embeddings.append(embedding)

    metadatas.append({
        "uri" : img_path,
        "name" : cls
    })

    ids.append(str(i))

print("Done!")

collection.add(
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)