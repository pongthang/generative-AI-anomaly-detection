## Import the models
import gradio as gr
import torch
from tqdm import tqdm
from monai.utils import set_determinism
from torch.cuda.amp import autocast
# from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet,AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
import cv2
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
PATH_auto = '/home/default/monai_generative_ai/auto_encoder_model.pt'

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
PATH_check_point = 'checkpoints/275.pth'
checkpoint = torch.load(PATH_check_point)

PATH_unet_condition = '/home/default/monai_generative_ai/unet_latent_space_model_condition.pt'
PATH_embed_condition = '/home/default/monai_generative_ai/embed_latent_space_model_condition.pt'

# unet.load_state_dict(torch.load(PATH_unet_condition))
# embed.load_state_dict(torch.load(PATH_embed_condition))

unet.load_state_dict(checkpoint['model_state_dict'])
embed.load_state_dict(checkpoint['embed_state_dict'])
####################################################################

unet.to(device)
embed.to(device)
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




with gr.Blocks() as demo:
    output =gr.Image(height=450,width=450)
    # output= gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Generate")
    input = gr.Radio(["Normal", "Level_1", "Level_2","Level_3","Worse"], label="Knee Osteoarthritis", info="Select the level of disease you want to generate !!")
    greet_btn.click(fn=generate_condition_bone_images, inputs=input,outputs=output, api_name="generate_bone")

if __name__ == "__main__":
    demo.launch(share=True,server_name='0.0.0.0')