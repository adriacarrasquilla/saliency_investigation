import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

original = "demo_outputs/org.png"
mask_out = "demo_outputs/mask_out.png"
infilled = "demo_outputs/infilled.png"

mask = (cv2.imread(mask_out, cv2.IMREAD_GRAYSCALE) > 0).astype(int)

with open("demo_outputs/title.txt", "r") as f:
    image_title = f.read()

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20,6))

imgs = [(axs[0], "Original", mpimg.imread(original)),
        (axs[1], "Masked out", mpimg.imread(original) * mask[:, :, np.newaxis]),
        (axs[2], "Infilled", mpimg.imread(infilled))]

for ax, title, img in imgs:
    ax.set(title=title, xticks=[], yticks=[])
    ax.set_axis_off()
    ax.imshow(img)

plt.suptitle(image_title, size=15)
plt.savefig("demo_outputs/complete_out.png")
plt.show()
