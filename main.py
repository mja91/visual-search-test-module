import os
import chromadb
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from glob import glob
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

model.to(device)

chroma_client = chromadb.Client()

collection = chroma_client.create_collection("test")

img_list = sorted(glob("test/*.jpg"))

embeddings = []
metadatas = []
ids = []

for i, img_path in enumerate(tqdm(img_list)):
    try:
        img = Image.open(img_path).convert("RGB")
        cls = os.path.split(os.path.dirname(img_path))[-1]

        img_tensor = image_processor(images=img, return_tensors="pt").to(device)
        outputs = model(**img_tensor)

        embedding = outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()

        embeddings.append(embedding)
        metadatas.append({
            "uri": img_path,
            "name": cls
        })
        ids.append(str(i))
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

if embeddings:
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

# 유사 이미지 검색 함수
def query(img_url, n_results=9):
    test_img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    test_img_tensor = image_processor(images=test_img, return_tensors="pt").to(device)
    test_outputs = model(**test_img_tensor)

    test_embedding = test_outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()

    query_embeddings = [test_embedding]

    query_result = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
    )

    fig, axes = plt.subplots(1, n_results + 1, figsize=(16, 10))

    axes[0].imshow(test_img)
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i, metadata in enumerate(query_result["metadatas"][0]):
        distance = query_result["distances"][0][i]
        img_path = metadata["uri"]

        try:
            retrieved_img = Image.open(img_path).convert("RGB")
            axes[i+1].imshow(retrieved_img)
            axes[i+1].set_title(f"{metadata['name']}: {distance:.2f}")
            axes[i+1].axis("off")
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")

    plt.show()

    return query_result

# fashion test query
query("https://image.msscdn.net/images/goods_img/20231016/3628385/3628385_16974223905869_500.jpg")
query("https://image.msscdn.net/images/goods_img/20210722/2037167/2037167_16941494688973_500.jpg")
query("https://image.msscdn.net/images/goods_img/20221220/2990516/2990516_16947563515196_500.jpg")
query("https://image.msscdn.net/images/goods_img/20210204/1778404/1778404_1_500.jpg")
query("https://image.msscdn.net/images/goods_img/20200818/1551840/1551840_1_500.jpg")
query("https://image.msscdn.net/images/goods_img/20201124/1699630/1699630_1_500.jpg")
