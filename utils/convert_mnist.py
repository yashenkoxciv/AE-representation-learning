import os
import argparse
from PIL import Image
from torchvision.datasets import MNIST
from tqdm import tqdm


def convert_mnist(root, train):
    mnist = MNIST(root, train=train, download=True)
    split_name = 'train' if train else 'test'
    for label in mnist.targets.unique().numpy():
        os.makedirs(
            os.path.join(root, split_name, str(label)),
            exist_ok=True
        )
    for i, (image, label) in tqdm(enumerate(zip(mnist.data.numpy(), mnist.targets.numpy()))):
        save_path = os.path.join(root, split_name, str(label), str(i) + '.jpg')
        Image.fromarray(image).save(save_path, 'JPEG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='root directory to work with MNIST')
    args = parser.parse_args()

    convert_mnist(root=args.root, train=True)
    convert_mnist(root=args.root, train=False)

