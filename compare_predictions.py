import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import casme.tasks.imagenet.utils as imagenet_utils
from casme.model_basics import casme_load_model, get_masks_and_check_predictions

from imagenet_labels import imagenet_labels

from PIL import Image
import cv2


def main():
    """
    TODO:
        - set idx as argument
        - Plot?
        - Save mask (binarized probably)
            - should we display multiple masks types?
        - test this with CAGAN
    """
    original = "demo_outputs/org.png"
    infilled = "demo_outputs/infilled.png"

    transform_actions = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
    ])


    model = casme_load_model("ca.chk", classifier_load_mode="pickled")

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

    imgs = [(axs[0], "Original", original, mpimg.imread(original)),
            (axs[1], "Infilled", infilled, mpimg.imread(infilled))]

    target = torch.tensor(0)
    for ax, title, path, img in imgs:
        ax.set_axis_off()
        ax.imshow(img)

        img_path = path
        image = Image.open(img_path)
        input_ = transform_actions(image)
        input_ = input_[None, :, :, :]

        _,_,_,_,_, output = get_masks_and_check_predictions(input_=input_, target=target, model=model, use_p=None)

        pred_idx = torch.argmax(output)
        predicted_class = imagenet_labels[int(pred_idx)]
        # print(img)
        # print(predicted_class)
        # print(torch.max(output))
        # print(torch.min(output))

        ax.set(title=f"{title}: {predicted_class}", xticks=[], yticks=[])

    plt.suptitle("Classification comparison", size=15)
    plt.savefig("demo_outputs/prediction_comp.png")
    plt.show()

if __name__ == "__main__":
    main()
