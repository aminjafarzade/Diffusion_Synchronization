import json
import subprocess

with open('/root/Diffusion-Assignment5-Synchronization/data/ambiguous_image_prompts.json', 'r') as file:
    data = json.load(file)


for key, value in data.items():

    command = ['python',
                'main.py',
                "--app",
                "ambiguous_image", 
                "--prompts",
                f"{value['canonical_prompt']}",
                f"{value['instance_prompt']}",
                "--views_names",
                "identity",
                "inner_rotate",
                "--tag",
                "ambiguous_image_inner_rotate",
                "--save_dir_now"]

    subprocess.run(command, capture_output=True, text=True, check=True)
