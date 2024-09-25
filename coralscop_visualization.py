import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import cv2
from matplotlib.patches import Polygon
import itertools
from visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
import argparse
color_mappings = {0: [224, 0, 0], 1: [138, 43, 226]}

for k in color_mappings:
    for i in range(3):
        color_mappings[k][i] /= 255.0

def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans

def unbunch_coords(coords):
    return list(itertools.chain(*coords))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns[:128]:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = color_mappings[1]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def add_boundary(node_coods,ax):
    polygon = Polygon(node_coods, closed=False, edgecolor='r')
    ax.add_patch(polygon)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="image path")
    parser.add_argument("--json_path", type=str, required=True, help="json path")
    parser.add_argument("--output_path", type=str, required=True, help="output path")
    parser.add_argument("--alpha", type=float, default=0.4, help="transparency")
    parser.add_argument("--min_area", type=float, default=4096, help="min area")
    args = parser.parse_args()
    alpha = args.alpha
    label_mode = '1'
    anno_mode = ['Mask']
    json_path = args.json_path
    img_path = args.img_path
    output_path = args.output_path
    min_area=args.min_area
    for files in glob.glob(os.path.join(json_path,"*.json")):
        with open(files, "r", encoding='utf-8') as f:
            aa = json.loads(f.read())
            images = aa['image']
            annotations = aa['annotations']
            img_name=images['file_name']
            print(img_name)
            _,file_name=os.path.split(img_name)
            if os.path.exists(os.path.join(output_path, file_name)):
                continue
            _,json_name=os.path.split(files)
            if not os.path.exists(os.path.join(img_path, json_name.replace(".json",".jpg"))):
                continue
            image = cv2.imread(os.path.join(img_path, json_name.replace(".json",".jpg")))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            visual = Visualizer(image, metadata=metadata)
            label = 1
            mask_map = np.zeros(image.shape, dtype=np.uint8)
            for i, ann in enumerate(annotations):
                if ann['segmentation']==[]:
                    continue
                mask = mask_util.decode(ann['segmentation'])
                if np.sum(mask)<min_area:
                    continue
                color_mask=color_mappings[1]
                demo = visual.draw_binary_mask_with_number(mask,color=color_mask,edge_color=[1.0,0,0], text="", label_mode=label_mode, alpha=alpha,
                                                           anno_mode=anno_mode)
                mask_map[mask == 1] = label
            im = demo.get_image()
            plt.figure(figsize=(20, 20))
            plt.imshow(im)
            plt.axis('off')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            plt.savefig(os.path.join(output_path,file_name),bbox_inches="tight")
            plt.gcf().clear()


if __name__ == "__main__":
    main()
