# The class containing the model
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNet:
    def __init__(self):
        # Source: https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt
        with open("src/imagenet_classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.eval()
        self.device = self.get_device()

    def get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def infer(self, image_path):
        input_image = Image.open(image_path)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        input_batch = input_batch.to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(output, 0)

        return (self.classes[index.item()], confidence.item())
