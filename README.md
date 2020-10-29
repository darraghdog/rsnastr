## RSNA-STR Pulmonary Embolism Detection

This is the code for the [RSNA-STR 2020 Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection) challenge.

![](figs/rsna_str.jpg?raw=true "Optional Title")


#### Preprocessing

Download the kaggle dataset and place in folder data, so we have a file `data/rsna-str-pulmonary-embolism-detection.zip`.  
  
For preprocessing we load the dicom file and window over each CT scan in the dicom, using the below windows,   
- RED channel / LUNG window / level=-600, width=1500  
- GREEN channel / PE window / level=100, width=700  
- BLUE channel / MEDIASTINAL window / level=40, width=400  
Each channel is stored in a differnt channel within a jpeg file to easy loading to models.  
If you would like more detail on windowing, check out [this descriptions](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930).

Run the following to create jpegs from impages. 

`nohup sh bin/run_01_prepare_data.sh &> logs/preprocess_run.out &`   

This will run in the background for around 4 hours. You can check how many files it was processed by `cat logs/preprocess_run.out | wc -l`.

This script runs a lot faster due to multithreading, however it does fail sometimes on some images. 
You can check how many failures you got by running `cat logs/preprocess_run.out | grep "Failed" | wc -l`.
It is recommended to rerun this a few times until the number of failures go to zero. A rerun will only process the previously failed images.  
You should see the number of processed and failed images drastically reducing in the 2nd and third run.   



