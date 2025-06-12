from PIL import Image
import os
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--distribution", choices=["normal", "uniform", "chi-sq", "lognormal"], required=True)
parser.add_argument("--timesteps", type=int, required=True)
parser.add_argument("--visual_dir", type=str, required=True)
parser.add_argument("--variables", nargs='+', required=True, help="List of variable names to visualize")
args = parser.parse_args()

# compile all plots from a channel over all timesteps into an animated gif
def sequence2gif(images_path, channel, num_frames, distribution):
    filenames = [f"{images_path}/frames/{distribution}_{channel}_t{t:02d}.png" for t in range(num_frames)]
    images = [Image.open(f) for f in filenames if os.path.exists(f)]
    
    output_path = os.path.join(images_path, f"animated_{distribution}_{channel}.gif")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # duration per frame in milliseconds
        loop=0
    )

for var in args.variables:
    sequence2gif(os.path.join(args.visual_dir, args.distribution, var), var, args.timesteps, args.distribution)