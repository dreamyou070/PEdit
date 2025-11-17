import numpy as np
import torch
import numpy as np
import torch
from diffusers.utils import load_image
from diffusers.models.embeddings import apply_rotary_emb
import numpy as np
import cv2
from diffusers.utils import load_image
from diffusers.models.embeddings import apply_rotary_emb
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
import torch
import math

def _to_numpy_rgb(img):
    # img: PIL.Image | np.ndarray | torch.Tensor | list[PIL]
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    if isinstance(img, np.ndarray):
        # possible shapes: (H,W,3) or (B,H,W,3) or (3,H,W)
        arr = img
        if arr.ndim == 4:  # take first in batch
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[-1] not in (1,3):
            # CHW -> HWC
            arr = np.transpose(arr, (1,2,0))
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:  # assume 0~255
            arr = arr / 255.0
        return arr
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 4:      # (B,C,H,W)
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1,3):  # (C,H,W) -> (H,W,C)
            t = t.permute(1,2,0)
        t = t.detach().cpu().float().numpy()
        if t.max() > 1.0:
            t = t / 255.0
        return t
    raise TypeError(f"Unsupported image type: {type(img)}")


def create_folder_with_full_permissions(path: str):
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o777)


def image_comparison(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compare two images using SSIM (Structural Similarity Index).
    Both images must be grayscale and have the same shape.
    """
    # Resize image2 to match image1
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    # SSIM expects float images in [0, 255] or normalized [0, 1]
    score = ssim(image1, image2, data_range=255)
    return score

def image_comparison_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0  # 특징점이 없는 경우

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    min_des = min(len(des1), len(des2))
    if min_des == 0:
        return 0.0

    similarity = len(good_matches) / min_des
    return similarity

@torch.no_grad()
def _to_range01(x):
    # 로깅/디버그용 보정이 필요할 때만 사용하세요 (학습 경로엔 넣지 마세요)
    return x.clamp(0, 1)

def latent2pil_qwen(self, latents):
    #with torch.no_grad():
    latents = self._unpack_latents(latents, 1024,1024, self.vae_scale_factor)
    latents = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = self.image_processor.postprocess(image, output_type='pil')[0]
    #    return image

def onetime_inference(self,
                      latents,
                      image_latents,
                      image_latents_4d,
                      t,
                      i,
                      guidance,
                      pooled_prompt_embeds,
                      prompt_embeds,
                      text_ids,
                      latent_ids,
                      num_warmup_steps=0,
                      progress_bar=None,
                      start_noise=None,
                      case_name=None,
                      **kwargs, ):

    self._current_timestep = t
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    save_time = int(timestep[0].round().item())
    def forward_once():

        def choose_y0_by_snr(
                pipe,  # pipe.vae (학습시 no_grad 금지!)
                start_noise,  # BCHW, same shape as latents
                noise_pred,  # BCHW
                latents,  # BCHW
                source_image,  # BCHW, [0,1] 가정 (torch tensor)
                save_time: int,
                eps: float = 1e-8,
        ):
            dtype = self.vae.dtype# $#latents.dtype
            self.vae = self.vae.to('cuda')
            if save_time > 300:
                x0_pred_latent = start_noise - noise_pred  # grad 흐름 유지
                x0_pred_latent = pipe._unpack_latents(x0_pred_latent, 1024, 1024, pipe.vae_scale_factor)
                x0_pred_latent = (x0_pred_latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                y_img = pipe.vae.decode(x0_pred_latent.to(dtype), return_dict=False)[0] # before processing
            else:
                xt_pred_latent = latents
                xt_pred_latent = pipe._unpack_latents(xt_pred_latent, 1024, 1024, pipe.vae_scale_factor)
                xt_pred_latent = (xt_pred_latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                y_img = pipe.vae.decode(xt_pred_latent.to(dtype), return_dict=False)[0]
            y_img_pil = pipe.image_processor.postprocess(y_img.detach(), output_type='pil')[0]
            ref_img_pil = pipe.image_processor.postprocess(source_image.detach(), output_type='pil')[0]
            ref = np.asarray(y_img_pil, dtype=np.float32) / 255.0
            img = np.asarray(ref_img_pil, dtype=np.float32) / 255.0
            mse = np.mean((img - ref) ** 2)

            sig = np.mean(ref ** 2)
            snr_db = 10.0 * math.log10((sig + eps) / (mse + eps))
            return y_img_pil, ref_img_pil, snr_db

        latent_model_input = torch.cat([latents, image_latents], dim=1)


        noise_pred = self.transformer(
            hidden_states=latent_model_input.to('cuda'),  # 이미 to() 완료
            timestep=timestep / 1000.0,
            guidance=guidance,  # 텐서면 to() 완료
            pooled_projections=pooled_prompt_embeds.to('cuda'),  # 이미 to() 완료
            encoder_hidden_states=prompt_embeds.to('cuda'),  # 이미 to() 완료
            txt_ids=text_ids,  # 이미 to() 완료
            img_ids=latent_ids,  # 이미 to() 완료
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]
        y0_pil, ref_img_pil, snr_src = choose_y0_by_snr(self, start_noise, noise_pred, latents, image_latents_4d,
                                           save_time= save_time)
        return noise_pred, y0_pil, ref_img_pil, snr_src

    noise_pred, y0_pil, ref_img_pil, snr_src = forward_once()
    latents_dtype = latents.dtype
    latents = latents.to(latents_dtype)
    metrics = {"step": int(i),"snr": snr_src,"case_name": case_name}
    return latents, y0_pil, metrics, noise_pred

def onetime_inference_qwen(self,
                           latents,
                           image_latents,
                           image_latents_4d,
                           start_noise,
                           t,
                           i,
                           case_name,
                           guidance,
                           encoder_hidden_states_mask,
                           encoder_hidden_states,
                           img_shapes,
                           txt_seq_lens,
                           ):

    self._current_timestep = t
    timestep = t.expand(latents.shape[0]).to('cuda', torch.bfloat16)
    save_time = int(timestep[0].round().item())

    def forward_once():

        def choose_y0_by_snr(
                pipe,  # pipe.vae (학습시 no_grad 금지!)
                start_noise,  # BCHW, same shape as latents
                noise_pred,  # BCHW
                latents,  # BCHW
                source_image,  # BCHW, [0,1] 가정 (torch tensor)
                save_time: int,
                eps: float = 1e-8,
        ):

            def latnet2pil(latents):

                self.vae.to('cuda')
                for name, param in self.vae.named_parameters():
                    param.to('cuda')
                latents = self._unpack_latents(latents, 1024, 1024, self.vae_scale_factor)
                latents = latents.to(self.vae.dtype)
                latents_mean = (torch.tensor(self.vae.config.latents_mean)
                                .view(1, self.vae.config.z_dim, 1, 1, 1)
                                .to(latents.device, latents.dtype))
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

                return  latents / latents_std + latents_mean

            self.vae = self.vae.to('cuda')
            org_dtype = latents.dtype
            if save_time > 300:
                y_img = latnet2pil(start_noise - noise_pred )
            else:
                y_img = latnet2pil(latents)
            y_img = y_img.to('cuda', dtype=org_dtype) # 128
            ref = source_image.to('cuda', dtype=org_dtype) # 1024
            mse = (y_img - ref).pow(2).mean()
            sig = (ref).pow(2).mean()
            snr_ratio = (sig + eps) / (mse + eps)
            snr_db = 10.0 * torch.log10(snr_ratio)
            return y_img, snr_db

        latent_model_input = torch.cat([latents, image_latents], dim=1)
        noise_pred = self.transformer(hidden_states=latent_model_input.to(torch.bfloat16),
            timestep=(timestep / 1000).to(torch.bfloat16),
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
            encoder_hidden_states_mask=encoder_hidden_states_mask.to(torch.bfloat16),
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs={},
            return_dict=False,)[0]
        noise_pred = noise_pred[:, : latents.size(1)].to(torch.bfloat16)
        y0_pil, snr_src = choose_y0_by_snr(self,
                                           start_noise,
                                           noise_pred,
                                           latents,
                                           image_latents_4d,
                                           save_time=save_time)
        return noise_pred, y0_pil, snr_src

    noise_pred, y0_pil, snr_src = forward_once()
    metrics = {"step": int(i),
               "snr": snr_src,
               "case_name": case_name, }
    return latents, noise_pred, metrics,y0_pil