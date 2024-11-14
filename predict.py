import torch
from PIL import Image

import argparse

import baseSet
from func_import import *

import dataPr
import ResNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=str)
    # print(parser.parse_args())
    return parser.parse_args()


def imgProcess(imgPath):
    imgTransfm=dataPr.imgTransfm(baseSet.picSize)
    imgOri=Image.open(imgPath)
    return imgTransfm(imgOri)

def tagRead():
    tagDict={}
    with open(baseSet.tagSave, "r", encoding='utf-8') as f_tags:
        data=f_tags.read()
        for line in data.split("\n"):
            if line:
                key, value=line.split('\t')
                tagDict[key]=value
    # print(tagDict)
    return tagDict

def predictFunc(tagNum, data):
    net=ResNet.modelLoad(tagNum, baseSet.stateSave, baseSet.device, 1)[0]
    pred=net(data).argmax(axis=1)
    return int(pred)

def main():
    args=get_args()

    imgPr=imgProcess(f"../../{args.image}")[None,].to(baseSet.device)
    # print(imgPr.shape)
    tagDict=tagRead()
    tagIndex=predictFunc(len(tagDict), imgPr)

    print(tagDict[str(tagIndex)])


if __name__=="__main__":
    main()
