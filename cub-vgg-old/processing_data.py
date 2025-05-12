import os
import random
import pickle
import argparse
from collections import defaultdict as ddict

def extract_data(data_dir, val_ratio = 0.2):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_dir + '/images')
    val_ratio = val_ratio

    path_to_id_map  = dict()
    with open(os.path.join(cwd, data_dir + '/images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[os.path.join(data_path, items[1])] = int(items[0])
        
        attribute_labels_all = ddict(list) # id - attribute list
        attribute_certainties_all = ddict(list) # id attribute certainties
        attribute_uncertain_labels_all = ddict(list) # id - calibrated attributes
        
        uncertainty_map = {
            1: {1: 0, 2: 0.5, 3: 0.8, 4: 1},
            0: {1: 0, 2: 0.5, 3: 0.2, 4: 0}
        }
        with open(os.path.join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
            for line in f:
                file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
                attribute_label = int(attribute_label)
                attribute_certainty = int(attribute_certainty)
                uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
                attribute_labels_all[int(file_idx)].append(attribute_label)
                attribute_certainties_all[int(file_idx)].append(attribute_certainty)
                attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)

        is_train_test = dict()
        with open(os.path.join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
            for line in f:
                items = line.strip().split()
                is_train_test[int(items[0])] = int(items[1])
        print("Number of training images: ", len([k for k, v in is_train_test.items() if v == 1]))
        print("Number of testing images: ", len([k for k, v in is_train_test.items() if v == 0]))
        print("Number of images: ", len(is_train_test))

        train_data, test_data = [], []
        folder_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        folder_list.sort()
        for i, folder in enumerate(folder_list):
            folder_path = os.path.join(data_path, folder)
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            for file in file_list:
                image_id = path_to_id_map[os.path.join(folder_path, file)]
                img_path = os.path.join(folder_path, file)
                metadata = {
                    'id': image_id,
                    'path': img_path,
                    'label': i,
                    'attr_label': attribute_labels_all[image_id],
                    'attr_certainty': attribute_certainties_all[image_id],
                    'uncertain_attr_label': attribute_uncertain_labels_all[image_id],
                }
                if is_train_test[image_id]:
                    train_data.append(metadata)
                else:
                    test_data.append(metadata)

        random.shuffle(train_data)
        split = int (val_ratio * len(train_data))
        val_data = train_data[:split]
        train_data = train_data[split:]
        print("Number of training images: ", len(train_data))
        print("Number of validation images: ", len(val_data))
        print("Number of testing images: ", len(test_data))
        # # Print sample metadata
        # print ("Sample metadata:")
        # print (train_data[0])
        return train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('-output', '-o', help='Where to save new datasets')
    parser.add_argument('-input', '-i', help='Where to load the datasets')
    args = parser.parse_args()
    train_data, val_data, test_data = extract_data(args.input)

    for dataset in ['train', 'val', 'test']:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        with open(os.path.join(args.output, f'{dataset}_data.pkl'), 'wb') as f:
            if dataset == 'train':
                pickle.dump(train_data, f)
            elif dataset == 'val':
                pickle.dump(val_data, f)
            else:
                pickle.dump(test_data, f)
    print(f"Data saved to {args.output}")
    