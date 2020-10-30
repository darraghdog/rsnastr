## RSNA-STR Pulmonary Embolism Detection

This is the code for the [RSNA-STR 2020 Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection) challenge.

#### Architecture

![](docs/architecture.jpg?raw=true "Optional Title")

#### Environment set up

The solution was tested on a [NVIDIA DGX A100](https://www.nvidia.com/en-us/data-center/dgx-a100/) with Ubuntu 18.04.4 LTS as operating system.   
The full solution runs end to end in approx 12 hours on a single A100 card. 
This card has 40GB GPU memory, if you have less memory, you can use lower batchsize, but increase accumulation to simulate the same batchsize used. 
It is important to keep the batchsize large due to the number of targets and imbalanced labels.

You are provided with a docker file. You can run without this by enuring the necessary packages are available. The docker environment can be set up like below (please ensure docker is installed).  
```
docker build -t rsnav04 -f DockerFile.docker .
docker run -itd --name RSNA_CONTAINERV04 -v $PWD:/mount --rm rsnav04:latest 
docker attach RSNA_CONTAINERV04
```
  
At minimum, the training and inference steps should be run from inside the docker container.  

#### Preprocessing

Download the competition dataset and place in folder data, so we have a file `data/rsna-str-pulmonary-embolism-detection.zip`.  
  
For preprocessing we load the dicom file and [window over each CT scan](https://github.com/darraghdog/rsnastr/blob/948d190422e4847229145ccfb09ad1d69ab6530c/preprocessing/dicom_to_jpeg.py#L32-L75) in the dicom, using the below windows,   
   
- RED channel / LUNG window / level=-600, width=1500  
- GREEN channel / PE window / level=100, width=700  
- BLUE channel / MEDIASTINAL window / level=40, width=400  
   
Each channel is stored in a differnt channel within a jpeg file to easy loading to models.  
If you would like more detail on windowing, check out [this description](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930).

Run the following to create jpegs from dicoms. 

`nohup sh bin/run_01_prepare_data.sh &> logs/preprocess_run.out &`   

This will run in the background for around 3 hours depending on your environment. There are a total of just under 2 million images to be extracted. You can check how many files were processed while the background process is running :  
`cat logs/preprocess_run.out | wc -l`  

This script runs a lot faster due to multithreading, however it does fail sometimes on some images. We run this a few times to pick up failed files. At the end, check the tail of the log to ensure no images are left.   
```
$ cat logs/preprocess_run.out | tail -4
2020-10-31 00:43:13,676 - Preprocess - INFO - Success train/13ef0b464626/8e86fda638ec/346da83668c7.dcm
2020-10-31 00:43:13,680 - Preprocess - INFO - Success train/fbe76b0deffe/6b26e3296bf3/81c2a6a8187a.dcm
2020-10-31 00:43:13,681 - Preprocess - INFO - Success train/f673adb91a15/578d5d64aae8/821cc62d3d2c.dcm
2020-10-31 00:43:43,020 - Preprocess - INFO - There are 0 unprocessed files
```

