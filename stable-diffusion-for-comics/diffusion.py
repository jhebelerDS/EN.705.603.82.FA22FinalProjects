import torch

import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

import argparse

parser = argparse.ArgumentParser(
    prog = "diffusion",
    description = "Stable Diffusion for Webcomic Art")

parser.add_argument('--prompt',
                    type=str,
                    default="Godzilla eating a meatball sub",
                    help="prompt to pass to stable diffusion")
parser.add_argument('--img',
                    type=str,
                    default=None,
                    help="pass a path to an image to switch to img2img mode")
parser.add_argument('--dream',
                    type=str,
                    default=None,
                    help="dreambooth help goes here")
parser.add_argument('--saveas',
                    type=str,
                    default='output',
                    help="what to name output file")

# img2img params 
parser.add_argument('--strength',
                    default=0.85,
                    help="How much to adhere to the original image (0,1)")
parser.add_argument('--guidance_scale',
                    default=7.5,
                    help="How much to adhere to the prompt")



if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda"

    # Determine Pipeline based on args
    if args.img == None and args.dream == None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16
        ).to(device)

        image = pipe(args.prompt).images[0]

    if args.img is not None:
        init_image = Image.open(args.img).convert("RGB")
        init_image = init_image.resize((768, 512))
        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "./stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
        ).to(device)

        image = pipe(prompt=args.prompt,
                     image = init_image,
                     strength = args.strength,
                     guidance_scale = args.guidance_scale).images[0]
        
    image.save(args.saveas+'.png')
