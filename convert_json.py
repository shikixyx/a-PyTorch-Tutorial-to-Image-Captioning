import json
from collections import defaultdict
from tqdm import tqdm

stair_captions_dir = "/Users/hiroaki/Documents/30_UT/04S/6_Enshu3/1_Miyao/Tutorial For Image Captioning/stair_captions_v1.2/"
train_tok = "stair_captions_v1.2_train_tokenized.json"
val_tok = "stair_captions_v1.2_val_tokenized.json"

data_coco = "/Users/hiroaki/Documents/30_UT/04S/6_Enshu3/1_Miyao/Tutorial For Image Captioning/caption_datasets/dataset_coco.json"


"""
images dataset

split:
test
restval

dict_keys(['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences', 'cocoid'])

{'filepath': 'val2014', 'sentids': [770337, 771687, 772707, 776154, 781998], 'filename': 'COCO_val2014_000000391895.jpg', 'imgid': 0, 'split': 'test', 'sentences': [{'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'], 'raw': 'A man with a red helmet on a small moped on a dirt road. ', 'imgid': 0, 'sentid': 770337}


{'image_id': 123127, 'tokenized_caption': 'Porte de Pantin 駅 に 電車 が 停 まっ て いる', 'id': 18, 'caption': 'Porte de Pantin駅に電車が停まっている'}
"""


with open(data_coco, "r") as f:
    data = json.load(f)

# 日本語キャプション
stair_captions = defaultdict(list)
for d in [train_tok, val_tok]:
    with open(stair_captions_dir + d, "r") as f:
        t = json.load(f)

        for img in t["annotations"]:
            cap = img["caption"]
            tok = img["tokenized_caption"].split()
            stair_captions[img["image_id"]].append((cap, tok))


# imageid -> [0]:raw [1]:token


L = len(data["images"])
for i in tqdm(range(L)):
    filename = data["images"][i]["filename"]
    imageid = filename.split("_")[2].split(".")[0]
    imageid = int(imageid)

    sentences = []
    for cap, tok in stair_captions[imageid]:
        dct = {"tokens": tok, "row": cap}
        sentences.append(dct)
    data["images"][i]["sentences"] = sentences


output_dir = "/Users/hiroaki/Documents/30_UT/04S/6_Enshu3/1_Miyao/Tutorial For Image Captioning/data.json"
with open(output_dir, "w") as f:
    json.dump(data, f, ensure_ascii=False)
