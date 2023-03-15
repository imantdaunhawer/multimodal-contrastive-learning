"""
Definition of datasets.
"""

import io
import json
import os
import numpy as np
import pandas as pd
import torch
from collections import Counter, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from torchvision.datasets.folder import pil_loader


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order of elements encountered."""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Multimodal3DIdent(torch.utils.data.Dataset):
    """Multimodal3DIdent Dataset.

    Attributes:
        FACTORS (dict): names of factors for image and text modalities.
        DISCRETE_FACTORS (dict): names of discrete factors, respectively.
    """

    FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            # 3: "object_zpos",  # is constant
            4: "object_alpharot",
            5: "object_betarot",
            6: "object_gammarot",
            7: "spotlight_pos",
            8: "object_color",
            9: "spotlight_color",
            10: "background_color"
        },
        "text": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            # 3: "object_zpos",  # is constant
            4: "object_color_index",
            5: "text_phrasing"
        }
    }

    DISCRETE_FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            # 3: "object_zpos",  # is constant
        },
        "text": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            # 3: "object_zpos",  # is constant
            4: "object_color_index",
            5: "text_phrasing"
        }
    }

    def __init__(self, data_dir, mode="train", transform=None,
            has_labels=True, vocab_filepath=None):
        """
        Args:
            data_dir (string): path to  directory.
            mode (string): name of data split, 'train', 'val', or 'test'.
            transform (callable): Optional transform to be applied.
            has_labels (bool): Indicates if the data has ground-truth labels.
            vocab_filepath (str): Optional path to a saved vocabulary. If None,
              the vocabulary will be (re-)created.
        """
        self.mode = mode
        self.transform = transform
        self.has_labels = has_labels
        self.data_dir = data_dir
        self.data_dir_mode = os.path.join(data_dir, mode)
        self.latents_text_filepath = \
            os.path.join(self.data_dir_mode, "latents_text.csv")
        self.latents_image_filepath = \
            os.path.join(self.data_dir_mode, "latents_image.csv")
        self.text_filepath = \
            os.path.join(self.data_dir_mode, "text", "text_raw.txt")
        self.image_dir = os.path.join(self.data_dir_mode, "images")

        # load text
        text_in_sentences, text_in_words = self._load_text()
        self.text_in_sentences = text_in_sentences   # sentence-tokenized text
        self.text_in_words = text_in_words           # word-tokenized text

        # determine num_samples and max_sequence_length
        self.num_samples = len(self.text_in_sentences)
        self.max_sequence_length = \
            max([len(sent) for sent in self.text_in_words]) + 1  # +1 for "eos"

        # load or create the vocabulary (i.e., word <-> index maps)
        self.w2i, self.i2w = self._load_vocab(vocab_filepath)
        self.vocab_size = len(self.w2i)
        if vocab_filepath:
            self.vocab_filepath = vocab_filepath
        else:
            self.vocab_filepath = os.path.join(self.data_dir, "vocab.json")

        # optionally, load ground-truth labels
        if has_labels:
            self.labels = self._load_labels()

        # create list of image filepaths
        image_paths = []
        width = int(np.ceil(np.log10(self.num_samples)))
        for i in range(self.num_samples):
            fp = os.path.join(self.image_dir, str(i).zfill(width) + ".png")
            image_paths.append(fp)
        self.image_paths = image_paths

    def get_w2i(self, word):
        try:
            return self.w2i[word]
        except KeyError:
            return "{unk}"  # special token for unknown words

    def _load_text(self):
        print(f"Tokenization of {self.mode} data...")

        # load raw text
        with open(self.text_filepath, "r") as f:
            text_raw = f.read()

        # create sentence-tokenized text
        text_in_sentences = sent_tokenize(text_raw)

        # create word-tokenized text
        text_in_words = [word_tokenize(sent) for sent in text_in_sentences]

        return text_in_sentences, text_in_words

    def _load_labels(self):

        # load image labels
        z_image = pd.read_csv(self.latents_image_filepath)

        # load text labels
        z_text = pd.read_csv(self.latents_text_filepath)

        # check if all factors are present
        for v in self.FACTORS["image"].values():
            assert v in z_image.keys()
        for v in self.FACTORS["text"].values():
            assert v in z_text.keys()

        # create label dict
        labels = {"z_image": z_image, "z_text": z_text}

        return labels

    def _create_vocab(self, vocab_filepath):
        print(f"Creating vocabulary as '{vocab_filepath}'...")

        if self.mode != "train":
            raise ValueError("Vocabulary should be created from training data")

        # initialize counter and word <-> index maps
        ordered_counter = OrderedCounter()  # counts occurrence of each word
        w2i = dict()  # word-to-index map
        i2w = dict()  # index-to-word map
        unique_words = []

        # add special tokens for padding, end-of-string, and unknown words
        special_tokens = ["{pad}", "{eos}", "{unk}"]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for i, words in enumerate(self.text_in_words):
            ordered_counter.update(words)

        for w, _ in ordered_counter.items():
            if w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unique_words.append(w)
        if len(w2i) != len(i2w):
            print(unique_words)
            raise ValueError("Mismatch between w2i and i2w mapping")

        # save vocabulary to disk
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(vocab_filepath, "wb") as vocab_file:
            jd = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(jd.encode("utf8", "replace"))

        return vocab

    def _load_vocab(self, vocab_filepath=None):
        if vocab_filepath is not None:
            with open(vocab_filepath, "r") as vocab_file:
                vocab = json.load(vocab_file)
        else:
            new_filepath = os.path.join(self.data_dir, "vocab.json")
            vocab = self._create_vocab(vocab_filepath=new_filepath)
        return (vocab["w2i"], vocab["i2w"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image
        img_name = self.image_paths[idx]
        image = pil_loader(img_name)
        if self.transform is not None:
            image = self.transform(image)

        # load text
        words = self.text_in_words[idx]
        words = words + ["{eos}"]
        words = words + ["{pad}" for c in range(self.max_sequence_length-len(words))]
        indices = [self.get_w2i(word) for word in words]
        indices_onehot = torch.nn.functional.one_hot(
            torch.Tensor(indices).long(), self.vocab_size).float()

        # load labels
        if self.has_labels:
            z_image = {k: v[idx] for k, v in self.labels["z_image"].items()}
            z_text = {k: v[idx] for k, v in self.labels["z_text"].items()}
        else:
            z_image, z_text = None, None

        sample = {
            "image": image,
            "text": indices_onehot,
            "z_image": z_image,
            "z_text": z_text}
        return sample

    def __len__(self):
        return self.num_samples
