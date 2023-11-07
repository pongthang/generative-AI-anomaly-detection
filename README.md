# Generative AI for Anomaly Detection for Bone Diseases

## Project Overview
Give me any knee x-ray image , I will tell you where there is arthritis. You can generate Knee x-ray of 5 different level of arthritis using this model. Train it in your own data and share it as medical dataset without fearing for privacy.


## Steps:

1) Vairational autoencoder (VAE) is train to convert input images (112x112) into a latent space of 3x28x28.
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

1) First download the dataset
2) Open [Latent_space_using_autoencoder.ipynb](./Latent_space_using_autoencoder.ipynb) . Train the autoencoder first. Then save the model
3) Open [generate_classifier_free_guaidance_bone_condition.ipynb](./generate_classifier_free_guaidance_bone_condition.ipynb) Train the diffusion model , enbedding models and save the models.
4) Use the models in [ui_for_generating_bone_image_with_condition.py](./ui_for_generating_bone_image_with_condition.py)

## UI for diffusion model to generate x-ray images of different level of Knee Arthritis.
Gradio based UI is presented here below you can run it in your localhost.

![Screenshot from 2023-11-06 00-47-07](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/ea73d47c-5611-4c03-93a9-cb7531665453)

![Screenshot from 2023-11-06 00-47-44](https://github.com/pongthang/generative-AI-anomaly-detection/assets/57061570/c40575a8-4c1a-41df-a871-0f7962f81582)


## Run the GUI for image generation:
You will need GPU for running this. 
Load the trained model properly in [ui_for_generating_bone_image_with_condition.py](./ui_for_generating_bone_image_with_condition.py)
```
python3 ui_for_generating_bone_image_with_condition.py
or
gradio ui_for_generating_bone_image_with_condition.py
```
Go to the url printed in the terminal
Enjoy your generative model !!.

# Note if you don't want to train the models drive link is sharing below:

## Next is anomaly detection !!

## Contributing

Contributions to this project are welcome. Feel free to submit bug reports, feature requests, or even pull requests to improve the functionality and usability of this tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Make sure to review and comply with the licensing terms when using or modifying this project for your own purposes.

If you have any questions or need further assistance with the project, please don't hesitate to reach out to the project maintainers.

**Thank you for using "Generative AI for Anomaly Detection for Bone Diseases"!**


