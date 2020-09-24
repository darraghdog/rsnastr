## RSNA-STR Pulmonary Embolism Detection

This is the code for the [RSNA-STR 2020 Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection) challenge.

#### To try
    Fix focal loss - and try with different pe_weights
    Add acumulation
    Gett RNN to check accuracy on negative pe
    Smooth labels over time
    Read up on other windowing approaches
    *Check if larger batch size helps/ Up lr with accumulation
    Add percentage seen for pos and neg and studies
    Save each of the best weights
    Try average over folds
    Fix scheduler
    Add fold to weights and logs
    Get a cycle up to test nmin and nmax

#### Results
| Model |Image Size|Epochs|Bag|TTA |Fold|ValSet|Val|LB|Config & comments                       |
| ---------------|----------|------|---|----|----|--------|------|--------|-------------------------|
| ResNeXt-101 32x8d|320|18|-|-|0|`5K-vestudy``5K+ve``5K-ve`|0.31283|-| Focal loss `configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json`|
| ResNeXt-101 32x8d|320|18|-|-|0|`5K-vestudy``5K+ve``5K-ve`|0.31687|-|`configs/_lr2308/rnxt101_lr1e4_binary.json` & Light aug|



![](figs/competition.png?raw=true "Optional Title")  
![](figs/scan.png) 
