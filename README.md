## End-to-End Supervised Lung Lobe Segmentation
In this project we present a fully automatic and supervised approach to the problem of the segmentation of the pulmonary lobes from a CT scan.
A 3D fully convolutional neural network was used based on the V-Net wich we called Fully Regularized V-Net (FRV-Net).
This work was performed in the Biomedical Imaging group at C-BER centre of INESC TEC, Portugal and it resulted in the paper "End-to-End Supervised Lung
Lobe Segmentation" accepted to the IJCNN2018 conference.
Here are the code and scripts to train our FRV-Net (as you select wich regularization techniques do you want) and to run the segmentations.


## Running a single segmentation with a pre-trained model.
To run a single segmentation with a pre-trained model a example file called "run_single_segmentation.py" is available. 
It teaches you how to open a CT scan, to open the model and to predict and save the segmentation.


## Train a model
If you want to train your model, a file called "train.py" is available.
It allows you to set the specific regularization techniques and parameters of the desired net.
	
	-path : Model path		  		(path)
	-train: Train data		  		(path)
	-val  : Validation data		 		(path)
	-lr   : Set the learning rate     		(float)
	-load : load a pre-trained model  		(boolean)
	-aux  : Multi-task learning	  		(float - weight in the loss function)
	-ds   : Number of Deep Supervisers		(int   - nº of layers)
	-bn   : Set Batch normalization  		(boolean)
	-dr   : Set Dropout              		(boolean)
	-fs   : Number of initial of conv channels	(int)

The train and validation datasets has to contain two folders A and B. where the folder A contains the CT scans and the B the correspondent ground-truth.
In the script file "train_session.sh", the examples used for our results are presented.

## Software
Our project was developed using Python (2.7) and Keras (2.0.4) framework that are required to use it.

