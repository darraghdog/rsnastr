## RSNA-STR Pulmonary Embolism Detection

This is the code for the [RSNA-STR 2020 Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection) challenge.

#### To try
    Try higher LR
    StepLR
    Fix ratio seen. 
    Smooth labels over time
    Read up on other windowing approaches
    Save each of the best weights
    Add fold to weights and logs
    Get a cycle up to test nmin and nmax

#### Results
|Image Size|Epochs|Bag|TTA |Fold|ValSet|Val|LB|Config & comments                       |
|----------|------|---|----|----|--------|------|--------|-------------------------|
|320|30|-|-|0 1 2 3 4|`5K-/5K+/5K-`|0.31309 0.32159 0.30927 0.30183 0.31173|-| Focal loss `configs/_lr2308/effnetb5_lr1e4_binary_focal_pe0.25.json`|
|320|18|-|-|0 1 2 3 4|`5K-/5K+/5K-`|0.31407 0.32609 0.30949 0.31444 0.31755|-| Focal loss `configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json`|
|320|22|-|-|0 1|`5K-/5K+/5K-`|0.32187 0.344201|-| Focal loss `configs/_lr2308/densenet169_lr1e4_binary_focal_pe0.25.json`|
|320|18|-|-|0|`5K-/5K+/5K-`|0.31687|-|`configs/_lr2308/rnxt101_lr1e4_binary.json` & Light aug|
|320|16|-|-|0|`5K-/5K+/5K-`|0.31833|-| Focal loss `configs/_lr2308/densenet201_lr1e4_binary_focal_pe0.25.json`|
|320|11|-|-|0|`5K-/5K+/5K-`|0.32505|-| Focal loss `configs/_lr2308/se101_lr1e4_binary_focal_pe0.25.json`|
|320|24|-|-|0|`5K-/5K+/5K-`|0.35697|-| Focal loss `configs/_lr2308/mixnet_xl_lr1e4_binary_focal_pe0.25.json`|


![](figs/competition.png?raw=true "Optional Title")  
![](figs/scan.png) 
