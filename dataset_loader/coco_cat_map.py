import json

path2val_json="/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/annotations_trainval2017/annotations/instances_val2017.json"
non_ordered_map_name = "/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/annotations_trainval2017/annotations/non_ordered_ids_to_ordered_map.json"
ordered_map_name = "/media/tamas.zsiros/64f02c55-245d-401c-9839-2ac782c0ae04/huggingface/coco/annotations_trainval2017/annotations/ordered_ids_to_non_ordered_map.json"

def get_ordered_map():
    with open(ordered_map_name, "r") as f:
        content = json.load(f)
    return content


def get_non_ordered_map():
    with open(non_ordered_map_name, "r") as f:
        content = json.load(f)
    return content


if __name__ == "__main__":
    with open(path2val_json, "r") as f:
        content = json.load(f)

    cats = content['categories']
    ordered_id_to_non_ordered_map = {}
    non_ordered_id_to_prdered_map = {}

    counter = 0

    for cat in cats:
        ordered_id_to_non_ordered_map[counter] = cat['id']
        non_ordered_id_to_prdered_map[cat['id']] = counter
        counter += 1

    with open(non_ordered_map_name, "w") as non_f:
        json.dump(non_ordered_id_to_prdered_map, non_f)

    with open(ordered_map_name, "w") as o_f:
        json.dump(ordered_id_to_non_ordered_map, o_f)


