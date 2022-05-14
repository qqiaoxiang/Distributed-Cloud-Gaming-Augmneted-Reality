## Distributed Cloud Gaming & Augmented Reality


> + Deep-sort: store object tracking code (Deepsort)
> + esrgan-tf2_1: prediction model of esrgan
> + model, urils, weights: yolo-v5 code, for object detection
> + AIDetector_pytorch.py: link yolo-v5 and tracking
> + saved_assets: Store the original image of each frame in the process of SR, the image after SR and the image after object detection

<u>_Install_</u>   
 
 1. git clone https://github.com/qqiaoxiang/Distributed-Cloud-Gaming-Augmneted-Reality.git #clone the repository
 
 2. cd Distributed-Cloud-Gaming-Augmented-Reality 
 
 3. pip install -r requirement.txt  
 
 4. python main.py --video_source <path of source video> --result_path <path to save video>
 
 <u>_Notes_</u>
The code needs to be executed in a GPU environment and the files for testing are provided here : gputest.py and torchtest.py 

### Relate Work
#### Uses keras library, and Pytorch, Tensorflow framework
