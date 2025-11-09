import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(Image.open("/Users/bhuvankumar/PycharmProjects/PythonProject2/data/A/00000.png"), cmap="gray")
ax[0].set_title("Original A")
ax[1].imshow(Image.open("/Users/bhuvankumar/PycharmProjects/PythonProject2/data/B/00000.png"), cmap="gray")
ax[1].set_title("A → B")
for a in ax: a.axis("off")
plt.show()
