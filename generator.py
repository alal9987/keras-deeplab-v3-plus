from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import cv2
from PIL import Image, ImageFile
import os
import settings


def encode_segmap(mask, n_classes):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(np.uint8)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        assert n_classes <= np.iinfo(np.uint8).max, "assert n_classes <= uint8 max"

        for ii, label in enumerate(settings.colors):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii

        #num_nonzero = np.count_nonzero(label_mask)
        #if num_nonzero > 0:
        #    print(num_nonzero)

        label_mask = to_categorical(label_mask, n_classes)

        return label_mask


class BatchGenerator(Sequence):

    def __init__(self, data_path, batch_size, mode='train', n_classes=3,
                 augment=False, num_aug=3, affine=False):

        self.images = []
        self.masks = []
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.is_augment = augment
        self.num_aug = num_aug
        self.affine = affine

        file_list = os.path.join(data_path, f'{mode}_test.txt')
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image, mask = line.split(',')
                image = os.path.join(data_path, image)
                mask = os.path.join(data_path, mask)
                assert os.path.isfile(image), f'File not found: {image}'
                assert os.path.isfile(mask), f'File not found: {mask}'
                self.images.append(image)
                self.masks.append(mask)
        print(f'{mode}| images: {len(self.images)}, masks: {len(self.masks)}')
        
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, i):
        images = []
        masks = []
        for n, (image_path, mask_path) in enumerate(zip(self.images[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.masks[i*self.batch_size:(i+1)*self.batch_size])):

            image = self.preprocess_image(image_path)
            mask = self.preprocess_mask(mask_path, self.n_classes)
            # TODO: augmentation
            # random horizontal flip
            # contrast
            # brightness
            # color
            # gaussian blur
            images.append(image / 255.0)
            masks.append(np.array(mask))

        images = np.asarray(images)
        masks = np.asarray(masks)

        #print('**', images.shape, masks.shape)

        return images, masks

    @staticmethod
    def preprocess_image(jpg_file):
        image = cv2.imread(jpg_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def preprocess_mask(png_file, n_classes):
        image = np.array(Image.open(png_file), dtype=np.uint8)

        h, w, c = image.shape
        assert c == 3, f"Invalid channel number: {c}. {png_file}"

        new_mask = encode_segmap(image, n_classes)

        #return Image.fromarray(new_mask)
        #print("**", new_mask.shape)
        return new_mask
