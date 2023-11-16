# Generative AI for Anomaly Detection for Bone Diseases

## Note: Aim is to build "Pathology AI" . 
Pathology AI is the application of various generative AI in healthcare especially in pathology. Pathology is a branch of medical science that is focused on the study and diagnosis of disease. Pathology AI will be a combination of diffusion models,vision transformers and LLM. This will revolutionize the world of healthcare.

## Project Overview
Give me any knee x-ray image , I will tell you where there is arthritis. You can generate Knee x-ray of 5 different level of arthritis using this model. Train it in your own data and share it as medical dataset without fearing for privacy.



## Steps:

1) Vairational autoencoder (VAE) is trained to convert input images (112x112) into a latent space of 3x28x28.
2) Diffusion model (DDPM) is tranined to generate x-ray images of different level of Knee Arthritis. Actually DDPM is trained in latent space and the autoencoder is used for encoding and decoding to grayscale image of size 112x112.
3) The same diffusion model is used to do image to image translation. A healthy image is generated from a given diseased image.
4) Comparsion between original diseased image and the generated healthy image as heat map.

## Install the required packages:
Make a conda environment for better experience.
1) Install pytorch for GPU . Follow the offical website for installing https://pytorch.org/get-started/locally
2) Install opencv. Follow the offical website for installing https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html 
3) Install monai , monai-generative , gradio and tqdm 
```
pip3 install monai==1.2.0  ## Errors comes due to different versio. Check the version
pip3 install monai-generative
pip3 install gradio
pip3 install tqdm
```
Additionally, please ensure you have a compatible version of MONAI installed. You can find more information on installing MONAI at [MONAI Installation Guide](https://github.com/Project-MONAI/MONAI#installation).


## How to train the model
I use Kaggle dataset - Knee Osteoarthritis Dataset with Severity Grading
https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity?resource=download
** GPU is needed for training.
1) First download the dataset
2) Open [Latent_space_using_autoencoder.ipynb](./Latent_space_using_autoencoder.ipynb) . Train the autoencoder first. Then save the model
3) Open [generate_classifier_free_guaidance_bone_condition.ipynb](./generate_classifier_free_guaidance_bone_condition.ipynb) Train the diffusion model , embedding models and save the models.
4) Open [anomaly_detection_knee.ipynb](./anomaly_detection_knee.ipynb) In this notebook anomaly detection is implemented. All the trained models are used here to generate healthy x-ray image from a diseased one.

** Change the paths of the trained models according to directory where you saved the models , when they are used in [anomaly_detection_knee.ipynb](./anomaly_detection_knee.ipynb) and [ui_main_anomaly.py](./ui_main_anomaly.py).

## Note: Pretrained model parameters - 
If you don't want to train or just want to test. Here is drive link . Download them
https://drive.google.com/drive/folders/165e5Bw253Y_8FTmKfgmP9x1WVeWVjmDz?usp=sharing 

## UI for diffusion model to generate x-ray images of different level of Knee Arthritis and anomaly detection.
Gradio based UI is presented here below you can run it in your localhost.
### Bone X-ray Image Generation:

![Screenshot from 2023-11-16 23-46-01](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/18c5b234-8088-42bd-8f92-dc056dfe9200)

#### Generated:

![Screenshot from 2023-11-16 23-47-02](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/2f76644e-9c8c-4319-9a3a-6ef3b58ec38b)

### Healthy image from given unhealthy image:

![Screenshot from 2023-11-16 23-44-58](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/fa67ba6f-2ad4-46cd-8a83-461602913f81)
### Generating anomaly map after adding contrast:

![Screenshot from 2023-11-16 23-45-40](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/d3d52e7b-8340-4e2b-ba26-16e1a490ee19)




## Run the above UI :
You will need GPU for running this. 
Load the trained models properly in [ui_main_anomaly.py](./ui_main_anomaly.py)
```
python3 ui_main_anomaly.py
or
gradio ui_main_anomaly.py
```
Go to the url printed in the terminal
Enjoy your generative model !!.

## Autoencoder:
* Input size - 112x112
* Latent space size - 3x28x28
![image](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/335c62ba-bd40-457a-a2cb-c54aedfe1d4e)


## Anomaly Detection.
From a given diseased x-ray image a healthy image is generated. We can think in other way like the model generate an x-ray image as if the same person doesn't have the disease.

![image](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/1546b239-a109-4f53-b699-7b01437f13e1)

## Difference between original and generated as heat-map.
![image](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/512617ba-ff4a-45b2-a302-4661b6ff14e0)

## Contributing

Contributions to this project are welcome. Feel free to submit bug reports, feature requests, or even pull requests to improve the functionality and usability of this tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Make sure to review and comply with the licensing terms when using or modifying this project for your own purposes.

If you have any questions or need further assistance with the project, please don't hesitate to reach out to the project maintainers.

**Thank you for using "Generative AI for Anomaly Detection for Bone Diseases"!**


