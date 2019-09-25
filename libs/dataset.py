# Originally written by skasai5296
# https://github.com/skasai5296/MSE


import numpy as np
import json
import os
import spacy
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class MSR_VTT_Features(Dataset):
    def __init__(
        self, dataset_dir, feature_dir, vocab, ann_file="videodatainfo_2017.json",
        hdf5_dir='hdf5', mode="train", align_size=(10, 7, 7)
    ):
        self.ft_dir = os.path.join(dataset_dir, feature_dir)
        self.hdf5_dir = os.path.join(dataset_dir, hdf5_dir)
        self.vocab = vocab

        # load spacy
        self.nlp = spacy.load("en")
        self.align_size = align_size
        self.video_ids = set()
        self.split_ids = set()
        self.data = []
        if mode == "train":
            begin, end = 0, 6513
        elif mode == "val":
            begin, end = 6513, 7010
        elif mode == "test":
            begin, end = 7010, 10000
        with open(os.path.join(dataset_dir, ann_file), "r") as f:
            ann = json.load(f)
            for video in ann["videos"]:
                # filter by mode
                if begin <= video["id"] < end:
                    self.split_ids.add(video["video_id"])
            for sentence in ann["sentences"]:
                video_id = sentence["video_id"]
                if video_id not in self.split_ids:
                    continue
                if video_id not in self.video_ids:
                    self.video_ids.add(video_id)
                    obj = {"video_id": int(video_id[5:]), "video_path": os.path.join(
                        self.ft_dir, video_id + ".pth"), "caption": [sentence["caption"]], "cap_id": [sentence["sen_id"]]}
                    self.data.append(obj)
                else:
                    self.data[
                        self.__len__() - 1]["caption"].append(sentence["caption"])
                    self.data[
                        self.__len__() - 1]["cap_id"].append(sentence["sen_id"])

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        id = data["video_id"]
        pth = data["video_path"]

        ft = torch.load(pth)
        if self.align_size:
            # interpolate features to given size
            if ft.size() != self.align_size:
                ft = F.interpolate(
                    ft.unsqueeze(0), size=self.align_size, mode='trilinear', align_corners=False).squeeze(0)

        # the list of captions
        caps = data["caption"]
        # randomly select one of captions
        cap_id = np.random.randint(0, len(caps))
        cap = caps[cap_id]

        # convert from token to tensor
        tokens = self.nlp(cap.lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token.text) for token in tokens])
        caption.append(self.vocab('<end>'))

        caption = torch.Tensor(caption)

        return {"feature": ft, "id": id, "caption": caption}


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (feature, caption)
            - feature: torch tensor of shape align_size.
            - caption: torch tensor of shape (?); variable length.
            - id: video id
    Returns:
        features: torch tensor of shape (batch_size, (align_size)).
        captions: torch tensor of shape (batch_size, padded_length).
        ids: list; video ids
        lengths: list; valid length for each padded caption.
    """

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x['caption']), reverse=True)

    feat_list = []
    ids = []
    cap_list = []
    for d in data:
        feat_list.append(d['feature'])
        ids.append(d['id'])
        cap_list.append(d['caption'])

    # merge features from tuple of 4D tensor to 5D tensor
    features = torch.stack(feat_list, dim=0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in cap_list]
    captions = torch.zeros(len(cap_list), max(lengths)).long()
    for i, cap in enumerate(cap_list):
        end = lengths[i]
        captions[i, :end] = cap[:end]

    return {"feature": features, "id": ids, "caption": captions, "length": lengths}
