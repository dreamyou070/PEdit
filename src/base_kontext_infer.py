import os
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont

def main():

    # [0] 기본 경로 설정
    base = '/MIR/nas1/dreamyou070/data'
    data_folder = os.path.join(base, 'EmuEdit', 'export', 'test', 'input_image')
    inst_folder = os.path.join(base, 'EmuEdit', 'export', 'test', 'edit_instruction')

    # call model
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    save_folder = f'/MIR/nas1/dreamyou070/Project/Project_EDIF/1_EmuEditBench/9_Kontext_Test'
    os.makedirs(save_folder, exist_ok=True)
    datas = os.listdir(data_folder)
    for data in datas:
        name, ext = os.path.splitext(data)
        source_path = os.path.join(data_folder, data)
        inst_path = os.path.join(inst_folder, f'{name}.txt')
        with open(inst_path, 'r') as f:
            inst = f.readlines()[0]

        input_image = Image.open(source_path)
        image = pipe(
            image=input_image,
            prompt=inst,
        ).images[0]
        image.save(os.path.join(save_folder, data))

if __name__ == '__main__':
    main()