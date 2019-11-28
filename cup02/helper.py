from tensorflow.keras import Model
import tensorflow as tf

import imgaug as ia
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

OUT_DIR = './dataset2/VOCdevkit_train/VOC2007/tmp/'
DATA_PATH = './dataset/pascal_voc_training_data.txt'
IMAGE_DIR = './dataset2/VOCdevkit_train/VOC2007/JPEGImages/'

# common params
NUM_CLASSES = 20
MAX_OBJECTS_PER_IMAGE = 20

class DatasetGeneratorXD():
    def __init__(self, data_path, image_dir, out_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        self.out_dir = out_dir
        self.image_names = []
        self.record_list = []
        self.object_num_list = []
        # filling the record_list
        input_file = open(self.data_path, 'r')
        self.fp = open(self.out_dir + "pascal_voc_training_data.txt", "w+")

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            self.image_names.append(ss[0])

            self.record_list.append([float(num) for num in ss[1:]])

            self.object_num_list.append(min(len(self.record_list[-1]) // 5, MAX_OBJECTS_PER_IMAGE))
            if len(self.record_list[-1]) < MAX_OBJECTS_PER_IMAGE * 5:
                self.record_list[-1] = self.record_list[-1] + \
                                       [0., 0., 0., 0., 0.] * \
                                       (MAX_OBJECTS_PER_IMAGE - len(self.record_list[-1]) // 5)
            elif len(self.record_list[-1]) > MAX_OBJECTS_PER_IMAGE * 5:
                self.record_list[-1] = self.record_list[-1][:MAX_OBJECTS_PER_IMAGE * 5]

    def _data_preprocess(self, image_name, raw_labels, object_num):
        # print(raw_labels)
        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        class_num = raw_labels[:, 4]
        xmin = xmin.numpy()
        ymin = ymin.numpy()
        xmax = xmax.numpy()
        ymax = ymax.numpy()
        class_num = class_num.numpy()

        image = imageio.imread(self.image_dir + image_name)

        b_list = []
        for i in range(object_num):
            b_list.append(BoundingBox(x1=xmin[i], x2=xmax[i], y1=ymin[i], y2=ymax[i]))
        bbs = BoundingBoxesOnImage(b_list, shape=image.shape)
        image_aug, bbs_aug = self._image_aug(image, bbs)

        image_before = bbs.draw_on_image(image, size=2)
        ia.imshow(image_before)

        #ia.imshow(image_aug)

        imageio.imsave(self.out_dir + image_name.split('.')[0] + '_gaussion.jpg', np.float32(image_aug))
        self.fp.write(image_name.split('.')[0] + '_gaussion.jpg')
        #print(image_name.split('.')[0] + '_gaussion.jpg')
        box_list = []
        for i in range(len(bbs.bounding_boxes)):
            after = bbs_aug.bounding_boxes[i]
            after_xmax = after.x1
            after_xmin = after.x1
            after_ymax = after.y1
            after_ymin = after.y1
            if after.x1 > after.x2:
                after_xmin = after.x2
            else:
                after_xmax = after.x2
            if after_xmin < 0:
                after_xmin = 0.1
            if after_xmax > image.shape[1]:
                after_xmax = image.shape[1] - 0.1

            if after.y1 > after.y2:
                after_ymin = after.y2
            else:
                after_ymax = after.y2
            if after_ymin < 0:
                after_ymin = 0.1
            if after_ymax > image.shape[0]:
                after_ymax = image.shape[0] - 0.1
            self.fp.write(' {} {} {} {} {}'.format(after_xmin, after_ymin, after_xmax,
                                                   after_ymax, class_num[i]))
            # print(' {} {} {} {} {}'.format(int(after_xmin), int(after_ymin), int(after_xmax),
            #                                int(after_ymax), int(class_num[i])))
            box_list.append([after_xmin, after_xmax, after_ymin, after_ymax])
        self.fp.write('\n')
        bf_list = []
        for i in (box_list):
            bf_list.append(BoundingBox(x1=i[0], x2=i[1], y1=i[2], y2=i[3]))
            bbs = BoundingBoxesOnImage(bf_list, shape=image.shape)
        image_after = bbs.draw_on_image(image_aug, size=2)
        ia.imshow(image_after)

    def _image_aug(self, image, bbs):
        seq = iaa.SomeOf((0, 3),[
            iaa.Noop(),
            iaa.Noop(),
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7))
                    ]),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                rotate=(-45, 45)
            ),
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.Grayscale(),
            iaa.Flipud(0.2)
        ], random_order=True)
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        rotate = iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
        image_aug2, bbs_aug2 = rotate(image=image_aug, bounding_boxes=bbs_aug)
        return image_aug2, bbs_aug2

    def generate(self):
        for i in range(len(self.image_names)):
            self._data_preprocess(self.image_names[i], np.array(self.record_list)[i], np.array(self.object_num_list)[i])
        self.fp.close()

if __name__ == "__main__":
    dg = DatasetGeneratorXD()
    dg.generate()
