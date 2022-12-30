docker run --gpus all -v $(pwd):/app --rm stable-diffusion python diffusion.py --prompt "a superhero flying over the city" --img "./source.jpeg" --saveas "./img2imgoutput.png"
