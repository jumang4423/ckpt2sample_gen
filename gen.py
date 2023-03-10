import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
import sys
import gradio
import random
from riffusion.img2audio import image_to_audio

def get_backend():
    cuda_flg = torch.cuda.is_available()
    mps_flg = torch.backends.mps.is_available()
    if cuda_flg:
        return "cuda"
    elif mps_flg:
        return "mps"
    else:
        return "cpu"

def genAudio(prompt, negative_prompt):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512, width=512,
    ).images[0]
    img_file = "./temp.png"
    audio_file = "./temp.wav"
    image.save(img_file)
    image_to_audio(image=img_file, audio=audio_file)

    return audio_file



# python gen.py <prompt string> <wav out filepath> <diffuser model path>
# args from command line
model_path = sys.argv[1]
print("model_path: ", model_path)

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    scheduler=scheduler
)
pipe = pipe.to(get_backend())
pipe.enable_attention_slicing()

demo = gradio.Interface(fn=genAudio, inputs=["text", "text"], outputs=["audio"])
demo.launch(share=True)
