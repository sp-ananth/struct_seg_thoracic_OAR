import tensorflow as tf
import cv2
import SimpleITK as sitk
import numpy as np


class SegDataLoader(object):

    def __init__(self,
                 image_list,
                 mask_list,
                 n_class,
                 batch_size=8,
                 image_size=(640, 640),
                 augment_rotate=True,
                 augment_crop_zoom=True,
                 augment_flip=True,
                 augment_color=True):

        self.image_list = image_list
        self.mask_list = mask_list
        self.n_class = n_class
        self.batch_size_slice = batch_size
        self.image_size = image_size
        self.augment_rotate = augment_rotate
        self.augment_crop_zoom = augment_crop_zoom
        self.augment_flip = augment_flip
        self.augment_color = augment_color

        self.data_len = len(image_list)
        assert len(self.image_list) == len(self.mask_list), "Data list and mask list do not match"
        self.all_paths = list(zip(image_list, mask_list))

    def load_data_label(self, image_mask_path, is_train=True):

        data_label = tf.py_func(return_random_slices, [image_mask_path, self.batch_size_slice, self.image_size],
                                tf.float32)
        data_label.set_shape([self.batch_size_slice, self.image_size[0], self.image_size[1], 2])

        if is_train:
            data_label = tf.map_fn(lambda x: self.preprocess_image_and_label(x[:, :, 0], x[:, :, 1]), data_label)
        data = data_label[:, :, :, 0]
        label = data_label[:, :, :, 1]
        data, label = self._postprocess_label(data, label)
        return data, label

    def preprocess_image_and_label(self, image, label):
        if self.augment_flip:
            distortions = tf.round(tf.random_uniform([2], dtype=tf.float32))
            distortions = tf.cast(distortions, tf.bool)
            image = self.flip_distortions(image, distortions=distortions)
            label = self.flip_distortions(label, distortions=distortions)
        if self.augment_rotate:
            random_angle = tf.random.uniform(shape=[1], minval=-np.pi / 4, maxval=np.pi / 4)
            image = self.rotate_image(image, random_angle)
            label = self.rotate_image(label, random_angle, is_label=True)
        if self.augment_crop_zoom:
            sample_distorted_bounding_box = self.random_zoom_box(image)
            image = self.crop_and_resize(image, sample_distorted_bounding_box)
            label = self.crop_and_resize(label, sample_distorted_bounding_box, is_label=True)
        if self.augment_color:
            image = self.color_distortions(image)

        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        return tf.stack([image, label], axis=-1)

    def color_distortions(self, image):
        image = tf.expand_dims(image, -1)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return tf.squeeze(image)

    def rotate_image(self, image, angle, is_label=False):
        if is_label:
            interpolation = 'NEAREST'
        else:
            interpolation = 'BILINEAR'
        return tf.contrib.image.rotate(image, angle, interpolation=interpolation)

    def flip_distortions(self, image, distortions):
        # Horizontal flipping
        distort_left_right_random = distortions[0]
        image = tf.cond(distort_left_right_random, lambda: tf.reverse(image, [1]), lambda: tf.identity(image))
        # Vertical flipping
        distort_up_down_random = distortions[1]
        image = tf.cond(distort_up_down_random, lambda: tf.reverse(image, [0]), lambda: tf.identity(image))
        return image

    def random_zoom_box(self, image):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            [self.image_size[0], self.image_size[1], 1],
            bounding_boxes=bbox,
            min_object_covered=0.65,
            area_range=[0.5, 1])
        return sample_distorted_bounding_box

    def crop_and_resize(self, image, sample_distorted_bounding_box, is_label=False):
        # Crop image
        image = tf.expand_dims(image, -1)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # set cropped image shape since the dynamic slice based upon the bbox_size loses the third dimension
        cropped_image.set_shape([None, None, 1])
        # Resize to original dimensions
        if is_label:
            interpolation = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            interpolation = tf.image.ResizeMethod.BILINEAR
        image = tf.image.resize_images(cropped_image, image.shape[:2], method=interpolation)
        return tf.squeeze(image)

    def _postprocess_label(self, data, label):
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(label, self.n_class, on_value=1.0, off_value=0.0)
        data = tf.expand_dims(data, axis=3)
        return data, label

    def get_train_batch(self, num_parallel_calls=4):
        with tf.name_scope('Train_data_loader'):
            # Apply tf data operations
            train_data = tf.data.Dataset.from_tensor_slices(self.all_paths)
            train_data = train_data.shuffle(buffer_size=self.data_len).repeat()
            train_data = train_data.map(lambda x: self.load_data_label(x, is_train=True),
                                        num_parallel_calls=num_parallel_calls)
            train_data = train_data.shuffle(buffer_size=self.batch_size_slice*3)
            train_data = train_data.prefetch(buffer_size=10)
            # Get batch from iterators
            train_batch_iterator = train_data.make_one_shot_iterator()
            train_batch_images, train_batch_labels = train_batch_iterator.get_next()
            return train_batch_images, train_batch_labels

    def get_eval_batch(self, num_parallel_calls=2):
        with tf.name_scope('Val_data_loader'):
            # Apply tf data operations
            eval_data = tf.data.Dataset.from_tensor_slices(self.all_paths)
            eval_data = eval_data.shuffle(buffer_size=self.data_len).repeat()
            eval_data = eval_data.map(lambda x: self.load_data_label(x, is_train=False),
                                      num_parallel_calls=num_parallel_calls)
            eval_data = eval_data.shuffle(buffer_size=self.batch_size_slice*3)
            eval_data = eval_data.prefetch(buffer_size=10)
            # Get batch from iterators
            eval_batch_iterator = eval_data.make_one_shot_iterator()
            eval_batch_images, eval_batch_labels = eval_batch_iterator.get_next()
            return eval_batch_images, eval_batch_labels


def return_random_slices(data_label_path, batch_size, image_size):

    def _normalize_planes(npzarray):
        maxHU, minHU = 400., -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
        return npzarray

    def _load_sitk_array(filename):
        image = sitk.ReadImage(filename)
        image = sitk.GetArrayFromImage(image)
        return image

    def _get_subsample_indices(label, background_label=0):
        selected_indices = list(
            map(lambda x: np.sum(x == np.ones_like(x) * background_label) != (x.shape[0] * x.shape[1]), label))
        return selected_indices

    def _normalize_slice(data_slice):
        data_slice = data_slice.astype(np.float32)
        return (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice))

    def _resize_slice(data_slice, label=False):
        if label:
            return cv2.resize(data_slice, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(data_slice, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

    # Load, normalize
    data = _load_sitk_array(data_label_path[0].decode())
    data = _normalize_planes(data)
    label = _load_sitk_array(data_label_path[1].decode())

    # Select informative indices
    selected_indices = _get_subsample_indices(label)
    data = data[selected_indices]
    label = label[selected_indices]

    # Shuffle, select batches
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = (data[idx])[:batch_size]
    label = (label[idx])[:batch_size]

    # Resize and normalize
    data = np.stack([_resize_slice(data_slice) for data_slice in data], axis=0)
    data = _normalize_slice(data)
    label = np.stack([_resize_slice(label_slice, label=True) for label_slice in label], axis=0)
    label = label.astype(np.float32)
    data_label = np.stack([data, label], axis=3)

    return data_label
