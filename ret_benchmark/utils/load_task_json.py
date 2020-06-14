import os
import json
import shutil
from collections import defaultdict


def load_task_json(task_json):
    with open(task_json, "r") as fr:
        infos = fr.readlines()

    img_task_dict = dict()

    counter = 0
    noisy_upc_list = list()
    for info in infos:
        data = json.loads(info)["data"]["imgs"]
        upc = json.loads(info)["data"]["upc"]
        if "metaKey" not in data[0]:
            counter += 1
            noisy_upc_list.append(upc)
        else:
            upc_dict = dict()
            for img in data:
                img_path = img["path"]
                is_clean = img["metaKey"]
                upc_dict[img_path] = is_clean
            img_task_dict[upc] = upc_dict
    return noisy_upc_list, img_task_dict


def vis_task_json(task_json):
    pass
