# InsKDCap
Knowledge-driven code for generating captions of insulator defect images.
## Installation and Requirements
1.Clone this repository and navigate to the folder
```bash
git clone https://github.com/SDS-model/InsKDCap.git
cd InsKDCap
```

2.Install necessary packages

Our code is modified based on the X-modaler framework. Please refer to [X-modaler](https://github.com/YehLi/xmodaler) for this part.

## Get Started

1.Data Preparation

1.1.Dataset Structure of Attribute-Aware Hash Code Mapping Network

This dataset adopts the following format.
```bash
{
    "texts": [
        "one string",
        "two strings",
        "three strings",
        "gray",
        "green",
        "white",
        "brown",
        "glass",
        "ceramic",
        "top",
        "middle",
        "left end",
        "right end",
        "bottom",
        "damage",
        "flashover",
        "self explosion"
    ],
    "groups": {
        "number of strings": [0, 1, 2],
        "color": [3, 4, 5,6],
        "material": [7,8],
        "defect location":[9,10,11,12,13],
        "defect type": [14, 15, 16]
      },
    "mappings": [
        {
            "visual_id": "1.npz",
            "text_ids": [1,5,7,13,16]  
        },
```
