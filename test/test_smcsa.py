import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

import torchvision.transforms as transforms
from PIL import Image
import torchvision
import torch
from hypothesis import given, strategies as st
from src.structure.ncsa import SegmentedMacrocolumn
import unittest
import numpy as np
from tqdm import tqdm

# np.random.seed(1111)
INPUT_SIZE = 512


def random_input(size):
    v = torch.rand((size,))
    return v


class TestSMCSA(unittest.TestCase):

    def test_consecutive_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input = random_input(512)
        output1 = m.run(input).flatten()
        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))
        assert hamming < len(output1)*.005

    def test_consecutive_diff_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input1 = random_input(512)
        output1 = m.run(input1).flatten()

        input2 = random_input(512)
        output2 = m.run(input2).flatten()

        input_hamming = sum(abs(input2-input1))
        output_hamming = sum(abs(output2-output1))

        assert output_hamming > sum(output1)*.8

    def test_intervaled_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02, lr=1.)

        input = random_input(512)
        output1 = m.run(input).flatten()

        for i in range(10):
            input_others = random_input(512)
            _ = m.run(input_others).flatten()

        output2 = m.run(input).flatten()

        hamming = sum(abs(output2-output1))

        assert hamming < len(output1)*.005

    def test_high_capacity_intervaled_same_input(self):
        m = SegmentedMacrocolumn(2048, 512, .02)

        input = random_input(512)
        output1 = m.run(input).flatten()

        for i in tqdm(range(1000)):
            input_others = random_input(512)
            _ = m.run(input_others).flatten()

        output2 = m.run(input).flatten()

        # TODO Consider making similarity checking an api call.
        hamming = sum(abs(output2-output1))

        assert hamming < len(output1)*.005

    # TODO Create a test case that tests similarity of representations
    # for two inputs from outputs of ResNet50 from similar and different images

    def test_resnet_output(self):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((400, 600)),
        ])
        images = []
        for f in Path('data/db').iterdir():
            with f.open('rb') as fp:
                images.append(transform(Image.open(fp)))
        images = torch.stack(images)
        images = images/255.
        device = 'cpu'
        images.to(device)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, device=device).to(device)
        with torch.no_grad():
            features = model.backbone(images)
        print(features)

        feature_size = len(features['3'][0].flatten())
        m = SegmentedMacrocolumn(2048, feature_size, .02)

        output1 = m.run(features['3'][0].flatten()).flatten()
        output2 = m.run((features['3'][1].flatten())).flatten()

        hamming = sum(abs(output2-output1))
        assert hamming < len(output1)*.02
