import os
import json
from multiprocessing import Pool
import argparse
import numpy as np
from tqdm import tqdm
import time

# convert coco format to yolo format
def convert(size, box, keypoints):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    kpt_sets = []

    for kpt_set in keypoints:
        kpt_set.copy()
        kpt_set[::3] = np.array(kpt_set[::3]) * dw
        kpt_set[1::3] = np.array(kpt_set[1::3]) * dh
        kpt_sets.append(kpt_set)
    return (x, y, w, h), list(np.concatenate(kpt_sets).flat)


def process_image(img_data):
    img, data, ana_txt_save_path, id_map, split = img_data
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    ana_txt_name = head + ".txt"
    with open(os.path.join(ana_txt_save_path, ana_txt_name), 'w') as f_txt:
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box, keypoints = convert((img_width, img_height), ann["bbox"], [ann["keypoints"], ann["foot_kpts"]]) # only keypoints and foot_kpts
                line = (id_map[ann["category_id"]], *box, *keypoints)
                f_txt.write(('%g ' * len(line)).rstrip() % line + '\n')
    return './images/%s/%s.jpg\n' %(split, head)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', default='coco_kpts/annotations/instances_val2017.json',type=str, help="input: coco format(json)")
    parser.add_argument('--save_path', default='coco_kpts/labels/val2017', type=str, help="specify where to save the output dir of labels")
    parser.add_argument('--split', default='train', type=str,
    help="specify train/val split")
    arg = parser.parse_args()

    # args and initial setup...
    json_file =   arg.json_path 
    ana_txt_save_path = arg.save_path  
    split = arg.split
    print("json file: ", json_file )
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    num_processes = 16

    # start timer
    start = time.time()
    
    
    pool = Pool(processes=num_processes)  # Adjust the number of processes based on your machine's capability
    img_data_list = [(img, data, ana_txt_save_path, id_map, split) for img in data['images']]
    list_file_lines = pool.map(process_image, img_data_list)
    
    with open(os.path.join(ana_txt_save_path, split+'.txt'), 'w') as list_file:
        for line in list_file_lines:
            list_file.write(line)

    # end timer
    end = time.time()

    # print time in hours and minutes
    print("Time taken: ", (end - start) / 60, "minutes")


if __name__ == '__main__':
    main()


    # Usage
    # val python ./utils/coco2yolo_mul.py --json_path /coco-pose/annotations/coco_wholebody_val_v1.0.json --save_path /dataset/labels/val --split val
    # train python ./utils/coco2yolo_mul.py --json_path /coco-pose/annotations/coco_wholebody_train_v1.0.json --save_path /dataset/labels/train --split train