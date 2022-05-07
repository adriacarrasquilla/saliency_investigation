import torch
from PIL import Image
import torchvision.transforms as transforms

from casme.model_basics import casme_load_model, get_masks_and_check_predictions
from casme.utils.torch_utils import ImagePathDataset
import casme.tasks.imagenet.utils as imagenet_utils
from casme import archs
from casme.tasks.imagenet.sanity_checks import save_fig, get_image_arr

from imagenet_labels import imagenet_labels

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys


"""
Cool idxs:
- 271: manta
- 140: shark
"""


def main():
    """
    TODO:
        - set idx as argument
        - Plot?
        - Save mask (binarized probably)
            - should we display multiple masks types?
        - test this with CAGAN
    """
    transform_actions = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
    ])

    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset.from_path(
            config_path="experiments/metadata/val.json",
            transform=transform_actions,
            return_paths=True,
        ),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=False
    )

    original_classifier = archs.resnet50shared(pretrained=True).eval().to(device)

    model = casme_load_model("ca.chk", classifier_load_mode="pickled")

    target = torch.tensor(0)
    input_ = None
    if len(sys.argv) > 2:
        img_path = sys.argv[2]
        target_class = -1
        path = [img_path]
        image = Image.open(img_path)
        input_ = transform_actions(image)
        input_ = input_[None, :, :, :]
    else:
        idx = 0 if len(sys.argv) <= 1 else int(sys.argv[1])

        for i, ((input_, target), path) in enumerate(data_loader):
            if i != idx:
                continue
            break

        target_class = imagenet_labels[int(target[0])]

    continuous, binarized, _, _, _, output = \
        get_masks_and_check_predictions(
            input_=input_, target=target, model=model,
            use_p=None,
        )

    pred_idx = torch.argmax(output)
    predicted_class = imagenet_labels[int(pred_idx)]

    if target_class == predicted_class or target_class == -1:
        title = f"Classified as {predicted_class}"
    else:
        title = f"Expected {target_class}, but classified as {predicted_class}"

    Image.open(path[0]).resize(binarized[0,0,:,:].shape).save("demo_outputs/org.png")
    Image.fromarray(binarized[0,0,:,:]*255).convert("RGB").save("demo_outputs/mask_out.png", "PNG")
    Image.fromarray((1-binarized[0,0,:,:])*255).convert("RGB").save("demo_outputs/mask_in.png", "PNG")

    with open("demo_outputs/title.txt", "w") as f:
        f.write(title)

    input = input_[0,:,:,:]
    img = get_image_arr(input)
    save_fig(
        img=img,
        mask=continuous,
        title=title,
        path="demo_outputs/mask.png",
    )

if __name__ == "__main__":
    main()
