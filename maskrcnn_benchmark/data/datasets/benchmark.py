import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from functools import reduce
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import (
    CharPolygons,
    SegmentationCharMask,
    SegmentationMask,
)
import matplotlib.pyplot as plt


class BenchmarkDataset(object):
    def __init__(self, use_charann, dataset_dir, data_subset, transforms=None, ignore_difficult=False):
        assert data_subset in ["pretrain", "train", "test", "val"]
        self.use_charann = use_charann
        self.dataset_dir = dataset_dir
        self.data_subset = data_subset
        self.transforms = transforms
        self.ignore_difficult = ignore_difficult
        if self.data_subset == "pretrain":
            gts_path = "annotations/full_pretrain.json"
        elif self.data_subset == "train":
            gts_path = "annotations/full_train.json"
        elif self.data_subset == "test":
            gts_path = "annotations/full_test.json"
        else:
            gts_path = "annotations/full_val.json"
        self.gts = json.load(open(os.path.join(self.dataset_dir, gts_path), "r"))
        is_filter = (self.ignore_difficult and (data_subset == "pretrain" or data_subset == "train"))
        self.gt_keys = self.parse_gt_keys(is_filter)
        self.min_proposal_size = 2
        self.char_classes = self.get_char_classes(os.path.join(self.dataset_dir, "dict.txt"))
        self.vis = True

    def parse_gt_keys(self, is_filter):
        if is_filter:
            gt_keys = []
            for gt_key in self.gts['annotation'].keys():
                has_positive = False
                gt = self.gts['annotation'][gt_key]
                for granularity in gt['annotations']:
                    for box in gt['annotations'][granularity]:
                        if box['anno_cat'] == 'standard' and box['ignore'] == 0:
                            is_no_valid_box = len(box['xywh_rect']) != 4 and len(box['quad']) != 8 and \
                                                (len(box['quad']) <= 0 or len(box['quad'])%2 != 0)
                            if not is_no_valid_box:
                                has_positive = True
                                break
                    if has_positive:
                        break
                if has_positive:
                    gt_keys.append(gt_key)
        else:
            gt_keys = list(self.gts['annotation'].keys())
        return gt_keys

    def get_char_classes(self, dict_path):
        lines = open(dict_path, "r").readlines()
        char_classes = ""
        for line in lines:
            char = line[0]
            if char != "\n":
                char_classes += char
        return char_classes

    def __getitem__(self, index):
        gt = self.gts['annotation'][self.gt_keys[index]]
        image_path = os.path.join(self.dataset_dir, gt['name'])
        imname = os.path.basename(image_path)
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        words, rect_boxes, charsbbs, segmentations, labels = self.load_gt(gt, height, width)
        if words[0] == "":
            use_char_ann = False
        else:
            use_char_ann = True
        if not self.use_charann:
            use_char_ann = False
        target = BoxList(
            rect_boxes[:, :4], img.size, mode="xyxy", use_char_ann=use_char_ann
        )
        if self.ignore_difficult:
            labels = torch.from_numpy(np.array(labels))
        else:
            labels = torch.ones(len(rect_boxes))
        target.add_field("labels", labels)
        masks = SegmentationMask(segmentations, img.size)
        target.add_field("masks", masks)
        char_masks = SegmentationCharMask(
            charsbbs, words=words, use_char_ann=use_char_ann, size=img.size, char_num_classes=len(self.char_classes)
        )
        target.add_field("char_masks", char_masks)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # if self.vis:
        #     import pdb; pdb.set_trace()
        #     new_im = img.numpy().copy().transpose([1, 2, 0]) + [
        #         102.9801,
        #         115.9465,
        #         122.7717,
        #     ]
        #     new_im = Image.fromarray(new_im.astype(np.uint8)).convert("RGB")
        #     mask = target.extra_fields["masks"].polygons[0].convert("mask")
        #     mask = Image.fromarray((mask.numpy() * 255).astype(np.uint8)).convert("RGB")
        #     if self.use_charann and len(target.extra_fields["char_masks"].chars_boxes) > 0:
        #         m, _ = (
        #             target.extra_fields["char_masks"]
        #             .chars_boxes[0]
        #             .convert("char_mask")
        #         )
        #         color = self.creat_color_map(37, 255)
        #         color_map = color[m.numpy().astype(np.uint8)]
        #         char = Image.fromarray(color_map.astype(np.uint8)).convert("RGB")
        #         char = Image.blend(char, new_im, 0.5)
        #     else:
        #         char = new_im
        #     new = Image.blend(char, mask, 0.5)
        #     img_draw = ImageDraw.Draw(new)
        #     for box in target.bbox.numpy():
        #         box = list(box)
        #         box = box[:2] + [box[2], box[1]] + box[2:] + [box[0], box[3]] + box[:2]
        #         img_draw.line(box, fill=(255, 0, 0), width=2)
        #     # new.save("./vis/char_" + imname)
        #     # print(gt)
        #     # plt.imshow(new)
        #     # plt.show()
        return img, target, imname

    def creat_color_map(self, n_class, width):
        splits = int(np.ceil(np.power((n_class * 1.0), 1.0 / 3)))
        maps = []
        for i in range(splits):
            r = int(i * width * 1.0 / (splits - 1))
            for j in range(splits):
                g = int(j * width * 1.0 / (splits - 1))
                for k in range(splits - 1):
                    b = int(k * width * 1.0 / (splits - 1))
                    maps.append([r, g, b])
        return np.array(maps)

    def load_gt(self, gt, height=None, width=None):
        words, rect_boxes, charsboxes, segmentations, labels = [], [], [], [], []
        annotated_chars = []
        for char_box in gt['annotations']['char']:
            if char_box['ignore'] == 0:
                annotated_chars.append(char_box['transcript'])
            else:
                annotated_chars.append("###")
        annotated_chars = np.array(annotated_chars)
        for granularity in gt['annotations']:
            for box in gt['annotations'][granularity]:
                if box['anno_cat'] != 'standard':
                    continue
                loc = self.get_location(box, True)
                word = box['transcript']
                if loc is None:
                    continue
                if box['ignore'] == 1:
                    if self.ignore_difficult:
                        min_x = min(loc[::2]) - 1
                        min_y = min(loc[1::2]) - 1
                        max_x = max(loc[::2]) - 1
                        max_y = max(loc[1::2]) - 1
                        rect_box = [min_x, min_y, max_x, max_y]
                        segmentations.append([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
                        rect_boxes.append(rect_box)
                        words.append(word)
                        labels.append(-1)
                    else:
                        continue
                else:
                    min_x = min(loc[::2]) - 1
                    min_y = min(loc[1::2]) - 1
                    max_x = max(loc[::2]) - 1
                    max_y = max(loc[1::2]) - 1
                    rect_box = [min_x, min_y, max_x, max_y]
                    segmentations.append([loc])
                    tindex = len(rect_boxes)
                    rect_boxes.append(rect_box)
                    words.append(word)
                    labels.append(1)
                    charbbs = []
                    is_full_charbbs = True
                    word_poly = Polygon(np.array(loc).reshape(-1, 2))
                    for c in word:
                        char_indices = np.where(annotated_chars == c)[0]
                        if len(char_indices) > 0:
                            iocs = []
                            for char_index in char_indices:
                                char_loc = self.get_location(gt['annotations']['char'][char_index], False)
                                char_poly = Polygon(np.array(char_loc).reshape(-1, 2))
                                intersection = word_poly.intersection(char_poly)
                                ioc = intersection.area / char_poly.area
                                iocs.append(ioc)
                            iocs = np.array(iocs)
                            max_iocs_index = np.argmax(iocs)
                            if iocs[max_iocs_index] > 0:
                                charbb = np.zeros((10,), dtype=np.float32)
                                charbb[:8] = np.array(char_loc)
                                charbb[8] = self.char2num(c)
                                charbb[9] = tindex
                                charbbs.append(charbb)
                            else:
                                is_full_charbbs = False
                                break
                        else:
                            is_full_charbbs = False
                            break
                    if is_full_charbbs and len(charbbs) > 0:
                        charsboxes.append(charbbs)
        num_boxes = len(rect_boxes)
        if num_boxes > 0:
            keep_boxes = np.zeros((num_boxes, 5))
            keep_boxes[:, :4] = np.array(rect_boxes)
            keep_boxes[:, 4] = range(
                num_boxes
            )
            if self.use_charann:
                return words, np.array(keep_boxes), charsboxes, segmentations, labels
            else:
                charbbs = np.zeros((10,), dtype=np.float32)
                if len(charsboxes) == 0:
                    for _ in range(len(words)):
                        charsboxes.append([charbbs])
                return words, np.array(keep_boxes), charsboxes, segmentations, labels
        else:
            words.append("")
            charbbs = np.zeros((10,), dtype=np.float32)
            return (
                words,
                np.zeros((1, 5), dtype=np.float32),
                [[charbbs]],
                [[np.zeros((8,), dtype=np.float32)]],
                [1]
            )

    def get_location(self, box, is_accept_poly):
        poly = box['poly']
        quad = box['quad']
        xywh_rect = box['xywh_rect']
        if not is_accept_poly:
            poly = []
        if len(poly) > 0 and len(poly)%2 == 0:
            # loc = poly
            loc = reduce(lambda x, y: x + y, poly)
        elif len(quad) == 8:
            loc = quad
        elif len(xywh_rect) == 4:
            x, y, w, h = xywh_rect
            loc = [x-w/2, y-h/2, x+w/2, y-h/2, x+w/2, y+h/2, x-w/2, y+h/2]
        else:
            loc = None
        return loc

    def char2num(self, c):
        num = self.char_classes.index(c)
        return num

    def __len__(self):
        return len(self.gt_keys)

