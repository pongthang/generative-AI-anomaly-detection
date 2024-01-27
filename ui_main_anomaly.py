import gradio as gr
import torch
from tqdm import tqdm
from monai.utils import set_determinism
from torch.cuda.amp import autocast
# from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet,AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from generative.networks.schedulers.ddim import DDIMScheduler
import cv2
from lib_image_processing.contrast_brightness_lib import controller
from lib_image_processing.removebg_lib import get_mask
import matplotlib.pyplot as plt
import numpy as np
set_determinism(42)
torch.cuda.empty_cache()

## Load autoencoder

device = torch.device("cuda")

autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256),
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=(False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
root_dir = "/home/default/monai_generative_ai/trained_model_anomaly_detection"
PATH_auto = f'{root_dir}/auto_encoder_model.pt'

autoencoderkl.load_state_dict(torch.load(PATH_auto))
autoencoderkl = autoencoderkl.to(device)

#### Load unet and embedings

embedding_dimension = 64
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
    with_conditioning=True,
    cross_attention_dim=embedding_dimension
)

embed = torch.nn.Embedding(num_embeddings=6, embedding_dim=embedding_dimension, padding_idx=0)

#### Load the Model here ##########################################################
# PATH_check_point = 'checkpoints/275.pth'
# checkpoint = torch.load(PATH_check_point)

PATH_unet_condition = f'{root_dir}/unet_latent_space_model_condition.pt'
PATH_embed_condition = f'{root_dir}/embed_latent_space_model_condition.pt'

unet.load_state_dict(torch.load(PATH_unet_condition))
embed.load_state_dict(torch.load(PATH_embed_condition))

# unet.load_state_dict(checkpoint['model_state_dict'])
# embed.load_state_dict(checkpoint['embed_state_dict'])
####################################################################

unet.to(device)
embed.to(device)


###---------------> Global variables for anomaly detection <------------------##

input_unhealthy = ''
output_healthy = ''

### ------------------------> Anomaly detection <-----------------------###########

scheduler_ddims = DDIMScheduler(num_train_timesteps=1000,schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

def get_healthy(un_img): # un_img is in range 0-255 but model takes in range 0-1. conversion is needed.
    global input_unhealthy
    global output_healthy
    
    img = cv2.resize(un_img,(112,112)) # resizing here
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    input_unhealthy = gray_image.copy()
    gray_image.resize(112,112,1)
    img_tensor = torch.from_numpy(gray_image*1.0)
    img_tensor = img_tensor.permute(2,0,1)
    img_tensor /= 255.
    img_tensor = img_tensor.float()
    input = img_tensor.reshape((1,1,112,112))
    z_mu, z_sigma = autoencoderkl.encode(input.to(device))
    z = autoencoderkl.sampling(z_mu, z_sigma)
    
    unet.eval()
    guidance_scale = 3.0
    total_timesteps = 1000
    latent_space_depth = int(total_timesteps * 0.5) 
    current_img = z
    current_img = current_img.float()
    scheduler_ddims.set_timesteps(num_inference_steps=total_timesteps)
    ## Ecodings
    scheduler_ddims.clip_sample = False
    class_embedding = embed(torch.zeros(1).long().to(device)).unsqueeze(1)
    progress_bar = tqdm(range(30))
    for i in progress_bar:  # go through the noising process
        t = i
        with torch.no_grad():
            model_output = unet(current_img, timesteps=torch.Tensor((t,)).to(current_img.device), context=class_embedding)
        current_img, _ = scheduler_ddims.reversed_step(model_output, t, current_img)
        progress_bar.set_postfix({"timestep input": t})
    
    latent_img = current_img
    ## Decoding
    conditioning = torch.cat([torch.zeros(1).long(), torch.ones(1).long()], dim=0).to(device)
    class_embedding = embed(conditioning).unsqueeze(1)
    
    progress_bar = tqdm(range(500))
    for i in progress_bar:  # go through the denoising process
        t = latent_space_depth - i
        current_img_double = torch.cat([current_img] * 2)
        with torch.no_grad():
            model_output = unet(
                current_img_double, timesteps=torch.Tensor([t, t]).to(current_img.device), context=class_embedding
            )
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        current_img, _ = scheduler_ddims.step(noise_pred, t, current_img)
        progress_bar.set_postfix({"timestep input": t})
    # torch.cuda.empty_cache()
    current_img_decode = autoencoderkl.decode(current_img)
    
    out_image = current_img_decode[0][0].to('cpu').detach().numpy()
    out_image = 255*out_image
    out_image = (out_image).astype('uint8')
    output_healthy = out_image.copy()
    return cv2.resize(out_image,(448,448))

##------------------> Anomaly detection , contrast and background removal <-------------------##

def update(brightness,contrast): ##def update(brightness,contrast,thr1,thr2):
    unhealthy_c = controller(input_unhealthy,brightness,contrast)
    healthy_c = controller(output_healthy,brightness,contrast)
    # unhealthy_remove_bg =  get_mask(unhealthy_c,thr1,thr2)
    # healthy_remove_bg = get_mask(healthy_c,thr1,thr2)
    # diff_img = unhealthy_remove_bg - healthy_remove_bg
    diff_img = unhealthy_c - healthy_c
    cmap = plt.get_cmap('inferno')
    diff_img_a = cmap(diff_img)
    diff_img = np.delete(diff_img_a, 3, 2)
    return cv2.resize(healthy_c,(448,448)),cv2.resize(diff_img,(448,448))



### --------------> Image generation <----------------------------##############



scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
# scale_factor = 0.943597137928009
# inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)


def get_value(grad):
    info_dict = {"Normal":1, "Level_1":2, "Level_2":3,"Level_3":4,"Worse":5}
    return info_dict[grad]

def generate_condition_bone_images(grad=0):
    grad_value = get_value(grad)
    unet.eval()
    scheduler.clip_sample = True
    guidance_scale = 3
    conditioning = torch.cat([torch.zeros(1).long(), grad_value * torch.ones(1).long()], dim=0).to(
        device
    )  # 2*torch.ones(1).long() is the class label for the UNHEALTHY (tumor) class
    class_embedding = embed(conditioning).unsqueeze(
        1
    )  # cross attention expects shape [batch size, sequence length, channels]
    scheduler.set_timesteps(num_inference_steps=1000)
    noise = torch.randn((1, 3, 28, 28))
    noise = noise.to(device)
    
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:
        with autocast(enabled=True):
            with torch.no_grad():
                noise_input = torch.cat([noise] * 2)
                model_output = unet(noise_input, timesteps=torch.Tensor((t,)).to(noise.device), context=class_embedding,)
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
        noise, _ = scheduler.step(noise_pred, t, noise)
    with torch.no_grad():
        noise = autoencoderkl.decode(noise)
        img = (noise[0][0].to('cpu')).numpy()
    return cv2.resize(img,(448,448))



##--------------------------------> UI <-----------------------------##

# with gr.Blocks() as demo:
#     gr.Markdown("Generate Bone x-ray Images and detect anomaly !!")
#     with gr.Tab("Generate Image on conditions"):
#         output =gr.Image(height=450,width=450)
#         # output= gr.Textbox(label="Output Box")
#         greet_btn = gr.Button("Generate")
#         input = gr.Radio(["Normal", "Level_1", "Level_2","Level_3","Worse"], label="Knee Osteoarthritis", info="Select the level of disease you want to generate !!")


#     with gr.Tab("Anomaly Detection"):
#         gr.Markdown("Generate healthy x-ray image and compare with the original to get the anomaly.")
#         with gr.Row():
#             image_input = gr.Image(height=450,width=450)
#             img_out_heal = gr.Image(height=450,width=450)
#         generate_healthy_button = gr.Button("Generate")

#         gr.Markdown("Generate Anomaly map")
#         with gr.Row():
#             # image_input = gr.Image()
#             image_output = [gr.Image(height=450,width=450),gr.Image(height=450,width=450)] # contrast and anomaly
#         update_anomaly_button = gr.Button("Update")
#         inputs_vlaues = [gr.Slider(0, 510, value=284, label="Brightness", info="Choose between 0 and 510"),
#                         gr.Slider(0, 254, value=234, label="Contrast", info="Choose between 0 and 254"),
#                         # gr.Slider(0, 50, value=7, label="Canny Threshold 1", info="Choose between 0 and 50"),
#                         # gr.Slider(0, 50, value=20, label="Canny Threshold 2", info="Choose between 0 and 50"),
#                 ]
        
#         # inputs_vlaues.append(image_input)
#     greet_btn.click(fn=generate_condition_bone_images, inputs=input,outputs=output, api_name="generate_bone")
#     generate_healthy_button.click(get_healthy,inputs=image_input,outputs=img_out_heal)
#     update_anomaly_button.click(update, inputs=inputs_vlaues, outputs=image_output)


my_theme = 'YenLai/Superhuman'


with gr.Blocks(theme=my_theme,title="Knee Predict") as demo:
    gr.Markdown(""" # Knee Predict 
    ## Generative AI for Anomaly Detection and Analysis for Bone Diseases - Knee Osteoarthritis """  )
    
    with gr.Tab("Generate Image on conditions"):
        gr.Markdown("#### Generate Knee X-ray images with condition. You can select the level of Osteoarthritis and click on generate . Then the AI will generate Knee X-ray image of the given condition.")
        with gr.Row():
            output =gr.Image(height=450,width=450)
            gr.Image(value="images/doc_bone.png",label="AI-Assisted Healthcare")
        # output= gr.Textbox(label="Output Box")
        gr.Markdown(" ### Select the level of disease severity you want to generate !!")
        input = gr.Radio(["Normal", "Level_1", "Level_2","Level_3","Worse"], label="Knee Osteoarthritis Disease Severity Levels",scale=1)
        with gr.Row():
            greet_btn = gr.Button("Generate",size="lg",scale=1,interactive=True)
            gr.Markdown()
            gr.Markdown()
        


    with gr.Tab("Anomaly Detection"):
        gr.Markdown("### From a given unhealthy x-ray image generate a healthy image keeping the size and other important features")
        with gr.Row():
            image_input = gr.Image(height=450,width=450,label="Upload your knee x-ray here")
            img_out_heal = gr.Image(height=450,width=450)
        with gr.Row():
            gr.Markdown()
            generate_healthy_button = gr.Button("Generate",size="lg")
            gr.Markdown()

        gr.Markdown("""### Generate Anomaly by comparing the healthy and unhealthy Knee x-rays
                    #### Click the update button to update the anomaly after changing the contrast and brightness. 
                     """)
        with gr.Row():
            # image_input = gr.Image()
            image_output = [gr.Image(height=450,width=450),gr.Image(height=450,width=450)] # contrast and anomaly
        with gr.Row():
            gr.Markdown()
            update_anomaly_button = gr.Button("Update",size="lg")
            gr.Markdown()
        inputs_vlaues = [gr.Slider(0, 510, value=284, label="Brightness", info="Choose between 0 and 510"),
                        gr.Slider(0, 254, value=234, label="Contrast", info="Choose between 0 and 254"),
                        # gr.Slider(0, 50, value=7, label="Canny Threshold 1", info="Choose between 0 and 50"),
                        # gr.Slider(0, 50, value=20, label="Canny Threshold 2", info="Choose between 0 and 50"),
                ]
        
        # inputs_vlaues.append(image_input)
        gr.Examples(examples='examples' , fn=get_healthy, cache_examples=True, inputs=image_input, outputs=img_out_heal)
    greet_btn.click(fn=generate_condition_bone_images, inputs=input,outputs=output, api_name="generate_bone")
    generate_healthy_button.click(get_healthy,inputs=image_input,outputs=img_out_heal)
    update_anomaly_button.click(update, inputs=inputs_vlaues, outputs=image_output)



if __name__ == "__main__":
    demo.launch(share=True,server_name='0.0.0.0')





