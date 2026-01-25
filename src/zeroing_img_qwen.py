import os
from PIL import Image
import torch
#from diffusers import QwenImageEditPipeline
from qwqen_pipelines import QwenImageEditPipeline

class Controller():

    def __init__(self):

        self.img_alpha_dict = {}
        self.txt_alpha_dict = {}

    def set_alpha(self, attn_idx, txt_alpha, img_alpha):

        self.txt_alpha_dict[attn_idx] = txt_alpha
        self.img_alpha_dict[attn_idx] = img_alpha

    def reset(self):
        self.img_alpha_dict = {}
        self.txt_alpha_dict = {}


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,):

    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def main() :

    print(f' step 1. LOAD QWEN EDIT PIPELINE')
    device = 'cuda'
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from pedit.base_utils.qwen_pipeline import QwenImageEditPlusPipeline
    model_path = f"OPPOer/Qwen-Image-Edit-2509-Pruning"
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.to('cuda')
    transformer = pipeline.transformer

    print(f' step 2. Transformer Analysis')


    import csv
    csv_dir = f'/MIR/nas1/dreamyou070/data/EmuEdit/export/validation.csv'
    image_folder = '/MIR/nas1/dreamyou070/data/EmuEdit/export/validation/images'
    required_cols = ["fname", "instruction", "task", "input_caption", "output_caption", "hash", "idx"]
    with open(csv_dir, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in required_cols if c not in header]
        if missing:
            raise ValueError(f"CSV에 필요한 열이 없습니다: {missing} / 실제 헤더: {header}")
        contents = []
        skipped = 0
        for row in reader:
            fname = row["fname"]
            img_path = os.path.join(image_folder, fname)
            if not os.path.isfile(img_path):
                skipped += 1
                continue
            item = {
                "image_path": img_path,
                "instruction": row["instruction"],
                "task": row["task"],
                "input_caption": row["input_caption"],
                "output_caption": row["output_caption"],
                "hash": row["hash"],
                "idx": int(row["idx"]) if row["idx"].isdigit() else row["idx"],
                "fname": fname}
            contents.append(item)

    save_edit_folder = '/MIR/nas1/dreamyou070/Project/Project_EDIF/Zeroing/Zeroing_Experiment_Qwepwn/image_zeroing'
    os.makedirs(save_edit_folder, exist_ok=True)

    for data in contents:
        fname = data['fname']
        name, _ = os.path.splitext(fname)

        unique_folder = os.path.join(save_edit_folder, name)
        if not os.path.exists(unique_folder):
            os.makedirs(unique_folder, exist_ok=True)

            instruction = data['instruction']
            task = data['task'].lower()
            src_img = Image.open(data['image_path']).resize((1024, 1024))

            if task == 'global' or task == 'style' or task == 'background' and not os.path.exists(data['image_path']):

                def ca_forward(attn, layer_idx, controller):

                    def forward(hidden_states: torch.FloatTensor,  # Image stream
                                encoder_hidden_states: torch.FloatTensor = None,  # Text stream
                                encoder_hidden_states_mask: torch.FloatTensor = None,
                                attention_mask=None,
                                image_rotary_emb=None,
                                ) -> torch.FloatTensor:
                        if encoder_hidden_states is None:
                            raise ValueError(
                                "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

                        seq_txt = encoder_hidden_states.shape[1]

                        # ============================================================================================================
                        pix_len = hidden_states.shape[1]
                        source_len = int(pix_len // 2)
                        img_condition = hidden_states[:, source_len:, :]
                        text_condition = encoder_hidden_states
                        img_alpha = controller.img_alpha_dict[layer_idx]
                        txt_alpha = controller.txt_alpha_dict[layer_idx]
                        hidden_states[:, source_len:, :] = hidden_states[:, source_len:, :] * img_alpha
                        encoder_hidden_states = encoder_hidden_states * txt_alpha
                        # print(f' shape of hidden_states: {hidden_states.shape}')

                        # ============================================================================================================
                        img_query = attn.to_q(hidden_states)
                        img_key = attn.to_k(hidden_states)
                        img_value = attn.to_v(hidden_states)

                        # Compute QKV for text stream (context projections)
                        txt_query = attn.add_q_proj(encoder_hidden_states)
                        txt_key = attn.add_k_proj(encoder_hidden_states)
                        txt_value = attn.add_v_proj(encoder_hidden_states)

                        # Reshape for multi-head attention
                        img_query = img_query.unflatten(-1, (attn.heads, -1))
                        img_key = img_key.unflatten(-1, (attn.heads, -1))
                        img_value = img_value.unflatten(-1, (attn.heads, -1))

                        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
                        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
                        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

                        # Apply QK normalization
                        if attn.norm_q is not None:
                            img_query = attn.norm_q(img_query)
                        if attn.norm_k is not None:
                            img_key = attn.norm_k(img_key)
                        if attn.norm_added_q is not None:
                            txt_query = attn.norm_added_q(txt_query)
                        if attn.norm_added_k is not None:
                            txt_key = attn.norm_added_k(txt_key)

                        # Apply RoPE
                        if image_rotary_emb is not None:
                            img_freqs, txt_freqs = image_rotary_emb
                            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

                        # Concatenate for joint attention
                        # Order: [text, image]
                        joint_query = torch.cat([txt_query, img_query], dim=1)
                        joint_key = torch.cat([txt_key, img_key], dim=1)
                        joint_value = torch.cat([txt_value, img_value], dim=1)

                        # Compute joint attention
                        joint_hidden_states = dispatch_attention_fn(
                            joint_query,
                            joint_key,
                            joint_value,
                            attn_mask=attention_mask,
                            dropout_p=0.0,
                            is_causal=False,
                            backend=attn.processor._attention_backend,
                        )

                        # Reshape back
                        joint_hidden_states = joint_hidden_states.flatten(2, 3)
                        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

                        # Split attention outputs back
                        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
                        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

                        # Apply output projections
                        img_attn_output = attn.to_out[0](img_attn_output)
                        if len(attn.to_out) > 1:
                            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
                        txt_attn_output = attn.to_add_out(txt_attn_output)
                        return img_attn_output, txt_attn_output

                    return forward

                for target_block_idx in range(60) :
                    attention_id = 0
                    controller = Controller()
                    for name, module in transformer.named_modules():
                        if module.__class__.__name__ == 'Attention':
                            txt_alpha = 1.0  # maybe more ?
                            img_alpha = 1.0
                            if attention_id == target_block_idx :
                                img_alpha = 0.0
                            controller.set_alpha(attention_id, txt_alpha, img_alpha)
                            module.forward = ca_forward(module, attention_id, controller)
                            attention_id += 1

                    edit_result = pipeline(src_img, instruction, num_inference_steps=50).images[0]
                    save_dir = os.path.join(unique_folder, f'image_{target_block_idx}_zeroing.png')
                    print(f' save on {save_dir}')
                    edit_result.save(save_dir)


                # non zeroing
                for target_block_idx in range(60):
                    attention_id = 0
                    controller = Controller()
                    for name, module in transformer.named_modules():
                        if module.__class__.__name__ == 'Attention':
                            txt_alpha = 1.0  # maybe more ?
                            img_alpha = 1.0
                            controller.set_alpha(attention_id, txt_alpha, img_alpha)
                            module.forward = ca_forward(module, attention_id, controller)
                            attention_id += 1

                    edut_result = pipeline(src_img, instruction, num_inference_steps=50).images[0]

                    edut_result.save(os.path.join(unique_folder, f'image_non_zeroing.png'))


if __name__ == '__main__':
    main()