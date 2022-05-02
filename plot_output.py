import matplotlib.pyplot as plt
import matplotlib.image as mpimg

original = "demo_outputs/org.png"
mask_out = "demo_outputs/mask_out.png"
infilled = "demo_outputs/infilled.png"

with open("demo_outputs/title.txt", "r") as f:
    image_title = f.read()

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20,6))

imgs = [(axs[0], "Original", mpimg.imread(original)),
        (axs[1], "Mask", mpimg.imread(mask_out)),
        (axs[2], "Infilled", mpimg.imread(infilled))]

for ax, title, img in imgs:
    ax.set(title=title, xticks=[], yticks=[])
    ax.set_axis_off()
    ax.imshow(img)

plt.suptitle(image_title, size=15)
plt.savefig("demo_outputs/complete_out.png")
plt.show()
