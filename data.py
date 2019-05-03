from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
import pickle
from itertools import islice
from torchvision import transforms


class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, 'mini-imagenet/compacted_datasets', 'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        # for each class
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                #image_data = np.array(image_data, dtype='float32')

                #image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]



class TieredImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        self.root = root
        self.partition = partition  # train/val/test
        #self.preprocess()
        self.data_size = [3, 84, 84]

        # load data
        self.data = self.load_dataset()

        # if not self._check_exists_():
        #     self._init_folders_()
        #     if self.check_decompress():
        #         self._decompress_()
        #     self._preprocess_()


    def get_image_paths(self, file):
        images_path, class_names = [], []
        with open(file, 'r') as f:
            f.readline()
            for line in f:
                name, class_ = line.split(',')
                class_ = class_[0:(len(class_)-1)]
                path = self.root + '/tiered-imagenet/images/'+name
                images_path.append(path)
                class_names.append(class_)
        return class_names, images_path

    def preprocess(self):
        print('\nPreprocessing Tiered-Imagenet images...')
        (class_names_train, images_path_train) = self.get_image_paths('%s/tiered-imagenet/train.csv' % self.root)
        (class_names_test, images_path_test) = self.get_image_paths('%s/tiered-imagenet/test.csv' % self.root)
        (class_names_val, images_path_val) = self.get_image_paths('%s/tiered-imagenet/val.csv' % self.root)

        keys_train = list(set(class_names_train))
        keys_test = list(set(class_names_test))
        keys_val = list(set(class_names_val))
        label_encoder = {}
        label_decoder = {}
        for i in range(len(keys_train)):
            label_encoder[keys_train[i]] = i
            label_decoder[i] = keys_train[i]
        for i in range(len(keys_train), len(keys_train)+len(keys_test)):
            label_encoder[keys_test[i-len(keys_train)]] = i
            label_decoder[i] = keys_test[i-len(keys_train)]
        for i in range(len(keys_train)+len(keys_test), len(keys_train)+len(keys_test)+len(keys_val)):
            label_encoder[keys_val[i-len(keys_train) - len(keys_test)]] = i
            label_decoder[i] = keys_val[i-len(keys_train)-len(keys_test)]

        counter = 0
        train_set = {}

        for class_, path in zip(class_names_train, images_path_train):
            img = pil_image.open(path)
            img = img.convert('RGB')
            img = img.resize((84, 84), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')
            if label_encoder[class_] not in train_set:
                train_set[label_encoder[class_]] = []
            train_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(images_path_train)))

        test_set = {}
        for class_, path in zip(class_names_test, images_path_test):
            img = pil_image.open(path)
            img = img.convert('RGB')
            img = img.resize((84, 84), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')

            if label_encoder[class_] not in test_set:
                test_set[label_encoder[class_]] = []
            test_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter " + str(counter) + " from "+str(len(class_names_test)))

        val_set = {}
        for class_, path in zip(class_names_val, images_path_val):
            img = pil_image.open(path)
            img = img.convert('RGB')
            img = img.resize((84, 84), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')

            if label_encoder[class_] not in val_set:
                val_set[label_encoder[class_]] = []
            val_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(class_names_val)))

        partition_count = 0
        for item in self.chunks(train_set, 20):
            partition_count = partition_count + 1
            with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_train_{}.pickle'.format(partition_count)), 'wb') as handle:
                pickle.dump(item, handle, protocol=2)

        partition_count = 0
        for item in self.chunks(test_set, 20):
            partition_count = partition_count + 1
            with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_test_{}.pickle'.format(partition_count)), 'wb') as handle:
                pickle.dump(item, handle, protocol=2)

        partition_count = 0
        for item in self.chunks(val_set, 20):
            partition_count = partition_count + 1
            with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_val_{}.pickle'.format(partition_count)), 'wb') as handle:
                pickle.dump(item, handle, protocol=2)



        label_encoder = {}
        keys = list(train_set.keys()) + list(test_set.keys())
        for id_key, key in enumerate(keys):
            label_encoder[key] = id_key
        with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)

        print('Images preprocessed')

    def load_dataset(self):
        print("Loading dataset")
        data = {}
        if self.partition == 'train':
            num_partition = 18
        elif self.partition == 'val':
            num_partition = 5
        elif self.partition == 'test':
            num_partition = 8

        partition_count = 0
        for i in range(num_partition):
            partition_count = partition_count +1
            with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_{}_{}.pickle'.format(self.partition, partition_count)), 'rb') as handle:
                data.update(pickle.load(handle))

        # Resize images and normalize
        for class_ in data:
            for i in range(len(data[class_])):
                image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
                image_resized = image2resize.resize((self.data_size[2], self.data_size[1]))
                image_resized = np.array(image_resized, dtype='float32')

                # Normalize
                image_resized = np.transpose(image_resized, (2, 0, 1))
                image_resized[0, :, :] -= 120.45  # R
                image_resized[1, :, :] -= 115.74  # G
                image_resized[2, :, :] -= 104.65  # B
                image_resized /= 127.5

                data[class_][i] = image_resized

        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))
        return data

    def chunks(self, data, size=10000):
        it = iter(data)
        for i in range(0, len(data), size):
            yield {k: data[k] for k in islice(it, size)}

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx



        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]