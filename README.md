# Incubit Project
You can only find how to use the code in this readme file.  
Please refer to the slides for more techical details.

## 0. Installation
Clone the repo to anywhere you like and denote it as ProjRoot

The following packages are required:  
detectron2,  
shapely,  
rdp,  
cv2,  
matplotlib,  
if there is any missing packages, try install it via pip or conda




## 1. Data preparation
Denote the root of the data as DataRoot, it should contains the following contents:

- DataRoot

    - raw
        - 0_0.png
        - ...
        
    - annotations
        - 0_0.png-annotated.json
        - ...
    
Then you should run the following lines to prepare the data

      python Data/preProcessData.py --dataRoot DataRoot

The DataRoot will be updated as:  

- DataRoot
    - raw
        - 0_0.png
        - ...  
    - annotations
        - 0_0.png-annotated.json
        - ...
    - Demo 
        - annoBatch
        - ...
    - Exp 
        - annoBatch
        - ...
    
In the Exp part, the whole data is splitted into training (80%) and testing (20%) sets, a model (ExpModel) is trained on training set and tested on testing set.  
In the Demo part, we train another model (DemoModel) on all available data. DemoModel is for you to infer your own data which may have a better performance then the ExpModel.

## 2. Train a model
### 2.0 pretrained model
If you are not interested in training a model, download the "Exp_output/model_final.pth" and "Demo_output/model_final.pth" from the link below and put them in "ProjRoot/Exp_output/model_final.pth" and "ProjRoot/Demo_output/model_final.pth" , then go to step 3.

https://drive.google.com/drive/folders/1Agoz7wD7Xb21x0H4rUO210aSgWvxQGEI?usp=sharing

### 2.1 training
In order to train ExpModel and DemoModel, please run the following command

        python Experiment/train.py --dataRoot DataRoot --taskName TaskName

where TaskName is either "Exp" or "Demo".

The model will be saved at "ProjRoot/Exp_output" or "ProjRoot/Demo_output"
### 2.2 performence evaluation
For the Exp part, a performence evaluation is conducted after training, you can find the result in terminal if you train by yourself or find in my slides.

### 2.3 testing set visualization
For the Exp part, I also add a visualization tool. Please run the following command

        python Experiment/inferAndVisTest.py  --dataRoot DataRoot --taskName Exp

You can find all visualization results in "DataRoot/Exp/visRes".

## 3. Inference on your own Data
In order to infer your own data, you should put your own data in "DataRoot/Demo/imgInfer", then run

        python Demo/demoOnfull.py  --dataRoot DataRoot --taskName Demo 

you can find the json results in "DataRoot/Demo/resInfer".

If you further want to visualize the results, please run

        python Demo/demoOnfull.py  --dataRoot DataRoot --taskName Demo --isVis 1
then, you can find the visualization in "DataRoot/Demo/visInfer".



