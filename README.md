# My Cassava Leaf Disease Classification Competition Code

This is the code I used for training models for the [Kaggle Cassava Leaf Disease Classification Challenge.](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)
To be in the same state as when I trained models with it, an `input` folder structured like this must be added:
```
input
├── 2019_2020_merged
│   ├── depth_maps
│   │   ├── 1000015157.jpg
│   │   ...
│   │   └── train-healthy-99.jpg
│   ├── train.csv
│   └── train_images
│       ├── 1000015157.jpg
        ...
│       └── train-healthy-99.jpg
├── label_num_to_disease_map.json
├── pytorch-image-models-master
└── sam-optimizer-pytorch
```

The contents of `2019_2020_merged/train_images/` and `2019_2020_merged/train.csv` are from [zzy990106's merged dataset dataset.](https://www.kaggle.com/zzy990106/cassava-merged-data)


`input/` also needs rwightman's [pytorch-image-models-master](https://github.com/rwightman/pytorch-image-models) and davda54's [SAM optimizer for pytorch.](https://github.com/davda54/sam)


The folder `2019_2020_merged/depth_maps` is populated using `generate_depth_maps.py`.

My inference notebook submitted to Kaggle is [here](https://www.kaggle.com/aaroswings/ensemble-inference-notebook).

Note that much of this code is in a prototype state! I used it for experiments and there is a lot of ugly branching due to quickly prototyping out different ideas. It was definitely not a software engineering project.
