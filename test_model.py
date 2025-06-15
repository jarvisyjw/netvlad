import torch
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as T

# def input_transform(image_size=None):
#     MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
#     if image_size:
#         return T.Compose([
#             T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize(mean=MEAN, std=STD)
#         ])
#     else:
#         return T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=MEAN, std=STD)
#         ])

# Load the model
model = torch.hub.load("jarvisyjw/netvlad", 'my_model')
model.eval().to(device='cuda' if torch.cuda.is_available() else 'cpu')

descs = []
# iterate over the image in toy_dataset
query_paths = Path("./toy_dataset/queries").glob("*.jpg")
db_paths = Path("./toy_dataset/database").glob("*.jpg")
q_descs = []
db_descs = []
to_tensor = T.ToTensor()
query_paths_list = list(query_paths)
db_paths_list = list(db_paths)

# transform = input_transform((224, 224))  # Assuming the model expects 224x224 images
with torch.inference_mode():
    for q_path in query_paths_list:
        img = to_tensor(Image.open(q_path).convert("RGB")).unsqueeze(0)  # Add batch dimension
        print(img.shape)
        desc = model(img.to(device='cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Output for {q_path.name}: {desc.shape}")
        q_descs.append(desc.cpu().numpy())
    for db_path in db_paths_list:
        img = to_tensor(Image.open(db_path).convert("RGB")).unsqueeze(0)  # Add batch dimension
        print(img.shape)
        desc = model(img.to(device='cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Output for {db_path.name}: {desc.shape}")
        db_descs.append(desc.cpu().numpy())  # Move to CPU for distance computation

# retrieve the output for the all query images

# query_paths_list = [str(path) for path in query_paths]
# db_paths_list = [str(path) for path in db_paths]
# print(query_paths_list)
# print(db_paths_list)
# db_paths_list = list(db_paths)

# use faiss to find the top1 match in the database
import faiss
index = faiss.IndexFlatL2(4096)  # Assuming the descriptor size is 512

for db_desc in db_descs:
    print(f"Database descriptor shape: {db_desc.shape}")
# print(np.array(db_descs).shape)

index.add(np.array(db_descs).reshape(-1, 4096))  # Reshape if necessary
_, predictions = index.search(np.array(q_descs).reshape(-1, 4096), 1)
# print the predictions
print(query_paths_list)
print(db_paths_list)
print(predictions)
for i, pred in enumerate(predictions):
    print(f"Query {i} top1 match in database: {pred[0]}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(query_paths_list[i]))
    plt.title(f"Query {i}")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(Image.open(db_paths_list[pred[0]]))
    plt.title(f"Top1 Match {pred[0]}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./toy_dataset/example_results/query_{i}_top1_match.png")  
    plt.tight_layout()
    plt.show()



# for i, q_desc in enumerate(q_descs):
#     print(f"Query {i} descriptor shape: {q_desc.shape}")
#     # find the top1 match in the database
#     dist = np.linalg.norm(q_desc - np.array([db_desc for db_desc in db_descs]), axis=0)
#     top1_idx = np.argmin(dist)
#     print(f"Top1 match for query {i}: Database image {top1_idx}.")
#     # print(f"Top1 match for query {i}: Database image {top1_idx} with distance {dist[top1_idx]}")
#     # visualize the top1 match
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(Image.open(query_paths_list[i]))
#     ax[0].set_title(f"Query {i}")
#     ax[1].imshow(Image.open(db_paths[top1_idx]))
#     ax[1].set_title(f"Top1 Match {top1_idx}")
#     plt.show()