# Distributed Cloud Gaming & Augmented Reality



> + **Deep-sort**: object tracking code
> 
> + **esrgan-tf2_1**: prediction model of SR
> 
> + **model, urils, weights**: object detection
> 
> + **AIDetector_pytorch.py**: link object detection and tracking
> 
> + **saved_assets**: Store the original image of each frame in the process of SR, the image after SR and the image after object detection


**Install** 
 1. `git clone https://github.com/qqiaoxiang/Distributed-Cloud-Gaming-Augmneted-Reality.git`  # clone the repository
 
 2. `cd Distributed-Cloud-Gaming-Augmented-Reality` 
 
 3. `pip install -r requirement.txt` # It contains all the required dependencies
 
 4. `python main.py --video_source <path of source video> --result_path <path to save video>`
 
 
**Notes**   
The code needs to be executed in a <u>GPU environment</u> and the files for testing are provided here : gputest.py and torchtest.py
 
The command: python gputest.py / torchtest.py
 
The code is able to run on version 2.8.0 of keras and Tensorflow, Pytorch 1.11.0, Python3.10 and recommended to run in the conda environment.

