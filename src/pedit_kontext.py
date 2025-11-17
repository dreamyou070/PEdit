import os
import sys
from base_utils.scheduling import FlowMatchEulerDiscreteScheduler as CustomFMEDS
from base_utils.custom_editing_inference import onetime_inference, onetime_inference_qwen
from types import MethodType
import argparse
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.adam import Adam
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import wandb
import open_clip
from dotenv import load_dotenv
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel, EMAModel
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils.torch_utils import is_compiled_module

from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
import bitsandbytes as bnb

# ---- Local modules
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file
from toolkit.extension import get_all_extensions_process_dict
from toolkit.prompt_utils import PromptEmbeds
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor,
load_dotenv()
sys.path.insert(0, os.getcwd())
os.environ['DISABLE_TELEMETRY'] = 'YES'
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)
accelerator = get_accelerator()
import copy
from torch import nn
from collections import OrderedDict
import os
from base_utils.controller import set_alpha_dict, set_alphas_for_case
from typing import Union, List, Optional
from base_utils import Controller
import numpy as np
import yaml
from diffusers import T2IAdapter, ControlNetModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader
import torch
import torch.backends.cuda
try:
    from huggingface_hub import HfApi
except Exception:
    raise
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.ema import ExponentialMovingAverage
from toolkit.embedding import Embedding
from toolkit.ip_adapter import IPAdapter
from toolkit.models.decorator import Decorator
from toolkit.network_mixins import Network
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit.config import get_config
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.stable_diffusion_model import StableDiffusion
from jobs.process import BaseTrainProcess
import torch.nn.functional as F
from toolkit.train_tools import get_torch_dtype, LearnableSNRGamma
import gc
from tqdm import tqdm

from toolkit.config_modules import SaveConfig, LoggingConfig, NetworkConfig, TrainConfig, ModelConfig, \
    EmbeddingConfig, DatasetConfig, preprocess_dataset_raw_config, AdapterConfig, GuidanceConfig, \
    validate_configs, \
    DecoratorConfig
from toolkit.logging_aitk import create_logger
from diffusers import FluxTransformer2DModel
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc
from accelerate import Accelerator
import transformers
import diffusers
from toolkit.util.get_model import get_model_class

def flush():
    torch.cuda.empty_cache()
    gc.collect()


_LOG_HEADER_PRINTED = False
_CSV_WRITER = None

def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class SDTrainer(BaseTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):

        super().__init__(process_id, job, config)
        self.accelerator: Accelerator = get_accelerator()
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        self.sd: StableDiffusion
        self.embedding: Union[Embedding, None] = None
        self.custom_pipeline = kwargs.get("custom_pipeline", None)  # **wargs 에서 가져 오기
        self.step_num = 0
        self.start_step = 0
        self.epoch_num = 0
        self.last_save_step = 0
        # start at 1 so we can do a sample at the start
        self.grad_accumulation_step = 1
        # if true, then we do not do an optimizer step. We are accumulating gradients
        self.is_grad_accumulation_step = False
        self.device = str(self.accelerator.device)
        self.device_torch = self.accelerator.device
        network_config = self.get_conf('network', None)
        if network_config is not None:
            self.network_config = NetworkConfig(**network_config)
        else:
            self.network_config = None
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        model_config = self.get_conf('model', {})
        self.modules_being_trained: List[torch.nn.Module] = []
        model_config['dtype'] = self.train_config.dtype
        self.model_config = ModelConfig(**model_config)

        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.logging_config = LoggingConfig(**self.get_conf('logging', {}))
        self.logger = create_logger(self.logging_config, config)
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler = None
        self.data_loader: Union[DataLoader, None] = None
        self.data_loader_reg: Union[DataLoader, None] = None
        self.trigger_word = self.get_conf('trigger_word', None)

        self.guidance_config: Union[GuidanceConfig, None] = None
        guidance_config_raw = self.get_conf('guidance', None)
        if guidance_config_raw is not None:
            self.guidance_config = GuidanceConfig(**guidance_config_raw)

        # store is all are cached. Allows us to not load vae if we don't need to
        self.is_latents_cached = True
        raw_datasets = self.get_conf('datasets', None)

        if raw_datasets is not None and len(raw_datasets) > 0:
            raw_datasets = preprocess_dataset_raw_config(raw_datasets)
        self.datasets = None
        self.datasets_reg = None
        self.dataset_configs: List[DatasetConfig] = []

        self.params = []

        # add dataset text embedding cache to their config
        if self.train_config.cache_text_embeddings:
            for raw_dataset in raw_datasets:
                raw_dataset['cache_text_embeddings'] = True

        if raw_datasets is not None and len(raw_datasets) > 0:
            for raw_dataset in raw_datasets:
                dataset = DatasetConfig(**raw_dataset)
                # handle trigger word per dataset
                if dataset.trigger_word is None and self.trigger_word is not None:
                    dataset.trigger_word = self.trigger_word
                is_caching = dataset.cache_latents or dataset.cache_latents_to_disk
                if not is_caching:
                    self.is_latents_cached = False
                if dataset.is_reg:
                    if self.datasets_reg is None:
                        self.datasets_reg = []
                    self.datasets_reg.append(dataset)
                else:
                    if self.datasets is None:
                        self.datasets = []
                    self.datasets.append(dataset)
                self.dataset_configs.append(dataset)

        self.is_caching_text_embeddings = any(
            dataset.cache_text_embeddings for dataset in self.dataset_configs
        )

        self.embed_config = None
        embedding_raw = self.get_conf('embedding', None)
        if embedding_raw is not None:
            self.embed_config = EmbeddingConfig(**embedding_raw)

        self.decorator_config: DecoratorConfig = None
        decorator_raw = self.get_conf('decorator', None)
        if decorator_raw is not None:
            if not self.model_config.is_flux:
                raise ValueError("Decorators are only supported for Flux models currently")
            self.decorator_config = DecoratorConfig(**decorator_raw)

        # t2i adapter
        self.adapter_config = None
        adapter_raw = self.get_conf('adapter', None)
        if adapter_raw is not None:
            self.adapter_config = AdapterConfig(**adapter_raw)
            # sdxl adapters end in _xl. Only full_adapter_xl for now
            if self.model_config.is_xl and not self.adapter_config.adapter_type.endswith('_xl'):
                self.adapter_config.adapter_type += '_xl'

        # to hold network if there is one
        self.network: Union[Network, None] = None
        self.adapter: Union[
            T2IAdapter, IPAdapter, ClipVisionAdapter, ReferenceAdapter, CustomAdapter, ControlNetModel, None] = None
        self.embedding: Union[Embedding, None] = None
        self.decorator: Union[Decorator, None] = None

        is_training_adapter = self.adapter_config is not None and self.adapter_config.train

        self.do_lorm = self.get_conf('do_lorm', False)
        self.lorm_extract_mode = self.get_conf('lorm_extract_mode', 'ratio')
        self.lorm_extract_mode_param = self.get_conf('lorm_extract_mode_param', 0.25)
        # 'ratio', 0.25)

        # get the device state preset based on what we are training
        self.train_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=False  # we ensure them later
        )

        self.get_params_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=True  # We check for grads when getting params
        )

        # fine_tuning here is for training actual SD network, not LoRA, embeddings, etc. it is (Dreambooth, etc)
        self.is_fine_tuning = True
        if self.network_config is not None or is_training_adapter or self.embed_config is not None or self.decorator_config is not None:
            self.is_fine_tuning = False

        self.named_lora = False
        if self.embed_config is not None or is_training_adapter:
            self.named_lora = True
        self.snr_gos: Union[LearnableSNRGamma, None] = None
        self.ema: ExponentialMovingAverage = None

        validate_configs(self.train_config, self.model_config, self.save_config, self.dataset_configs)

        do_profiler = self.get_conf('torch_profiler', False)
        self.torch_profiler = None if not do_profiler else torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )

        self.current_boundary_index = 0
        self.steps_this_boundary = 0
        self.num_consecutive_oom = 0
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None
        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None
        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"
        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False
        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
            self.do_prior_prediction = True

        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")

    def before_model_load(self):
        pass

    def hook_before_model_load(self):
        # override in subclass
        pass

    def hook_after_sd_init_before_load(self):
        pass


    def run(self, args):

        BaseTrainProcess.run(self)
        d = self.dataset_configs[0]
        src_folder = d.src_folder
        inst_folder = d.inst_folder
        result_folder = d.result_folder
        os.makedirs(result_folder, exist_ok=True)
        # ==============================================================================
        max_epochs = d.max_epochs + args.max_epochs_add
        stage_1 = d.stage_1 + args.stage_1_add
        # ==============================================================================
        target_file_idx = d.file_idx + args.target
        for file_idx, filename in enumerate(os.listdir(src_folder)):
            if file_idx  == target_file_idx :
                name, _ = os.path.splitext(filename)
                unique_folder = os.path.join(result_folder, f'{name}')
                if not os.path.exists(unique_folder):

                    self.hook_before_model_load()
                    model_config_to_load = copy.deepcopy(self.model_config)
                    ModelClass = get_model_class(self.model_config)
                    if hasattr(ModelClass, 'get_train_scheduler'):
                        sampler = ModelClass.get_train_scheduler()
                    else:
                        arch = 'sd'
                        if self.model_config.is_flux:
                            arch = 'flux'
                        sampler = get_sampler(self.train_config.noise_scheduler,{"prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon", },arch=arch, )

                    if self.train_config.train_refiner and self.model_config.refiner_name_or_path is not None and self.network_config is None:
                        previous_refiner_save = self.get_latest_save_path(self.job.name + '_refiner')
                        if previous_refiner_save is not None:
                            model_config_to_load.refiner_name_or_path = previous_refiner_save
                            self.load_training_state_from_metadata(previous_refiner_save)
                    self.sd = ModelClass(device=self.accelerator.device,
                                         model_config=model_config_to_load,
                                         dtype=self.train_config.dtype,
                                         custom_pipeline=self.custom_pipeline,
                                         noise_scheduler=sampler, )
                    self.hook_after_sd_init_before_load()
                    self.sd.load_model()
                    device = 'cuda'
                    self.sd.pipeline.to(device)
                    self.sd.pipeline.vae.requires_grad_(False)
                    self.sd.pipeline.text_encoder_2.requires_grad_(False)
                    self.sd.pipeline.text_encoder.requires_grad_(False)
                    self.sd.pipeline.transformer.requires_grad_(True)
                    self.sd.pipeline.safety_checker = None
                    transformer = self.sd.pipeline.transformer
                    dtype = get_torch_dtype(self.train_config.dtype)
                    transformer.to('cuda')
                    for name, param in transformer.named_parameters():  #
                        param.to('cuda')
                        param.requires_grad_(False)
                    transformer.train()
                    if self.train_config.xformers:
                        self.sd.pipeline.vae.enable_xformers_memory_efficient_attention()
                        self.sd.pipeline.transformer.enable_xformers_memory_efficient_attention()
                        if isinstance(self.sd.pipeline.text_encoder, list):
                            for te in self.sd.pipeline.text_encoder:
                                if hasattr(te, 'enable_xformers_memory_efficient_attention'):
                                    te.enable_xformers_memory_efficient_attention()

                    if self.train_config.gradient_checkpointing:
                        if hasattr(self.sd.pipeline.transformer, 'enable_gradient_checkpointing'):
                            self.sd.pipeline.transformer.enable_gradient_checkpointing()
                        elif hasattr(self.sd.pipeline.transformer, 'gradient_checkpointing'):
                            self.sd.pipeline.transformer.gradient_checkpointing = True
                        else:
                            print("Gradient checkpointing not supported on this model")
                        if isinstance(self.sd.pipeline.text_encoder, list):
                            for te in self.sd.pipeline.text_encoder:
                                if hasattr(te, 'enable_gradient_checkpointing'):
                                    te.enable_gradient_checkpointing()
                                if hasattr(te, "gradient_checkpointing_enable"):
                                    te.gradient_checkpointing_enable()
                        else:
                            if hasattr(self.sd.pipeline.text_encoder, 'enable_gradient_checkpointing'):
                                self.sd.pipeline.text_encoder.enable_gradient_checkpointing()
                            if hasattr(self.sd.pipeline.text_encoder, "gradient_checkpointing_enable"):
                                self.sd.pipeline.text_encoder.gradient_checkpointing_enable()

                    if isinstance(self.sd.pipeline.text_encoder, list):
                        for te in self.sd.pipeline.text_encoder:
                            te.requires_grad_(False)
                            te.eval()
                    else:
                        self.sd.pipeline.text_encoder.requires_grad_(False)
                        self.sd.pipeline.text_encoder.eval()
                    self.sd.pipeline.transformer.to(self.device_torch, dtype=dtype)
                    self.sd.pipeline.vae = self.sd.pipeline.vae.to(torch.device('cpu'), dtype=dtype)
                    self.sd.pipeline.vae.requires_grad_(False)
                    self.sd.pipeline.vae.eval()

                    print(" step 1. pipeline customize with selt forward and scheduling")
                    self.sd.pipeline.enable_attention_slicing()
                    self.sd.pipeline.enable_vae_slicing()
                    self.sd.pipeline.enable_vae_tiling()
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    self.sd.pipeline.onetime_inference = MethodType(onetime_inference, self.sd.pipeline)

                    def patch_scheduler_step(scheduler):
                        if getattr(scheduler, "_orig_step", None) is not None:
                            return  # 이미 패치됨
                        scheduler._orig_step = scheduler.step

                        def _safe_sigma_index(self, sigma_val):
                            s = sigma_val.detach().float().item() if torch.is_tensor(sigma_val) else float(sigma_val)
                            sig = self.sigmas
                            if torch.is_tensor(sig):
                                idx = int(torch.argmin((sig - s).abs()).item())
                            else:
                                import numpy as np
                                idx = int(np.argmin(np.abs(np.asarray(sig) - s)))
                            return idx

                        def custom_step(self, model_output, sigma_t, sample, *args, **kwargs):
                            idx = _safe_sigma_index(self, sigma_t)
                            if idx >= len(self.sigmas) - 1:
                                idx = len(self.sigmas) - 2
                                sigma_t = self.sigmas[idx]
                            return self._orig_step(model_output, sigma_t, sample, *args, **kwargs)

                        scheduler.step = MethodType(custom_step, scheduler)

                    scheduler = self.sd.pipeline.scheduler
                    old_cfg = self.sd.pipeline.scheduler.config
                    new_sched = CustomFMEDS.from_config(old_cfg)
                    self.sd.pipeline.scheduler = new_sched
                    patch_scheduler_step(scheduler)

                    print(" step 2. Seed")
                    seed = 42
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    print(" step 3. call controller")
                    controller = Controller()

                    name, ext = os.path.splitext(filename)
                    out_name = name
                    src_path = os.path.join(src_folder, filename)
                    src_img = Image.open(src_path).convert("RGB").resize((1024, 1024))
                    txt_path = os.path.join(inst_folder, f"{name}.txt")
                    with open(txt_path, "r") as f:
                        editing_prompt = f.readlines()[0].strip()
                    print(f'editing_prompt: {editing_prompt}')
                    print(" step 5. valid len")
                    def get_valid_embeddinglength(prompt, pipe):
                        with torch.no_grad():
                            text_encoder = self.sd.pipeline.text_encoder_2
                            text_encoder.to('cuda')
                            encoded = self.sd.pipeline.tokenizer_2(
                                prompt,
                                padding="max_length",  # 파이프라인이 보통 이 옵션으로 고정해서 쓰고 있을 가능성 큼
                                truncation=True,
                                return_tensors="pt")
                            input_ids = encoded["input_ids"]  # [1, seq_len]
                            attention_mask = encoded["attention_mask"]  # [1, seq_len] (1은 유효, 0은 pad)
                            text_outputs = text_encoder(input_ids=input_ids.to(self.sd.pipeline.text_encoder_2.device),
                                                        attention_mask=attention_mask.to(self.sd.pipeline.text_encoder_2.device), )
                            del text_encoder
                            text_embeds = text_outputs.last_hidden_state
                        valid_len = int(attention_mask.sum().item())
                        return valid_len, text_embeds

                    valid_len, _ = get_valid_embeddinglength(editing_prompt, self.sd.pipeline)
                    controller.valid_len = valid_len
                    print(f' make unique_folder = {unique_folder}')
                    os.makedirs(unique_folder, exist_ok=True)

                    src_img.save(os.path.join(unique_folder, f"{out_name}_src.png"))
                    record_dir = os.path.join(unique_folder, 'timewise_score.txt')
                    with open(record_dir, "w") as ftxt:
                        ftxt.write(f'editing prompt : {editing_prompt}\n')
                    with open(record_dir, "a") as ftxt:
                        ftxt.write(f'cfg_name,tca_ratio,snr_score\n')
                    num_steps = 28
                    if d.data == 'HQ' :
                        configs = [("case1",1,1),("case2",8, 0), ("case3", 0, 8)]
                    else :
                        configs = [("case1", 1, 1), ("case2", 8, 0), ("case3", 0,0.1)]
                    model_name = 'Kontext'
                    config_dictionary = {}

                    for config in configs:
                        config_name, s1, s2 = config
                        config_dictionary[config_name], training_models = set_alpha_dict(model_name,
                                                                                         self.sd.pipeline.transformer,
                                                                                         config_name,
                                                                                         s1,
                                                                                         s2,
                                                                                         data=d.data)
                        if config_name == 'case3' :
                            case3_trainings = training_models
                        elif config_name == 'case2':
                            case2_trainings = training_models

                    print(" step 7. config setup")
                    device_ref = 'cuda' #pipe.transformer.device
                    dtype_ref = self.sd.pipeline.transformer.dtype  #
                    (base_latents,prompt_embeds,pooled_prompt_embeds,timesteps,src_latents,guidance,text_ids,
                     latent_ids,image_latents_4d) = self.sd.pipeline.prepare_inference(num_inference_steps=num_steps,
                                                                                       prompt=editing_prompt,
                                                                                       image=src_img)
                    lat_map_cpu = {cfg_name: base_latents.detach().clone().to("cpu") for cfg_name in config_dictionary}
                    pipe = self.sd.pipeline
                    controller.state = 'save'
                    device = 'cuda'
                    search = True
                    logging_dir = os.path.join(unique_folder, f"SNR_SCORE.TXT")
                    with open(logging_dir, "a") as ftxt:
                        ftxt.write(f'stage_idx,epoch,snr_score,tca_ratio\n')
                    import time
                    start_time = time.time()
                    with pipe.progress_bar(total=num_steps) as progress_bar:
                        for stage_idx, t in enumerate(timesteps):
                            result_check = []
                            def do_stage1(lat_map_cpu,t,src_latents,image_latents_4d,pooled_prompt_embeds, base_latents):
                                controller.stage = 'stage1'
                                casewise_embedding_dict = {}
                                timewise_metrics = {}
                                noise_pred_dict = {}
                                for cfg_name in config_dictionary:
                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name != 'case1':
                                        with torch.no_grad():
                                            set_alphas_for_case(pipe.transformer,
                                                                'Kontext',
                                                                controller,
                                                                cfg_name,
                                                                config_dict)
                                            _, y0_pil, metrics, noise_pred = self.sd.pipeline.onetime_inference(
                                                latents=lat_map_cpu[cfg_name].to(device, non_blocking=True).to('cuda'),
                                                image_latents=src_latents.to(device),
                                                image_latents_4d=image_latents_4d.to(device),
                                                t=t,
                                                i=stage_idx,
                                                guidance=guidance,
                                                start_noise=base_latents.to(device_ref, non_blocking=True),
                                                pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                                                prompt_embeds=prompt_embeds.to(device),
                                                text_ids=text_ids,
                                                latent_ids=latent_ids,
                                                timesteps=timesteps)
                                            if isinstance(y0_pil, Image.Image):
                                                y0_pil.save(os.path.join(unique_folder,f"{cfg_name}_{stage_idx}.png"))
                                            noise_pred_dict[cfg_name] = noise_pred.detach()
                                            embedding_dictionary = controller.embedding_dictionary
                                            casewise_embedding_dict[cfg_name] = embedding_dictionary
                                            controller.embedding_dictionary = {}
                                            tcas = 0
                                            icas = 0
                                            tca_dict = controller.tca_dict
                                            ica_dict = controller.ica_dict
                                            for layer in tca_dict.keys():
                                                tcas += tca_dict[layer]
                                                icas += ica_dict[layer]
                                            tca_ratio = tcas / (tcas + icas)
                                            controller.reset()
                                            controller.tca_dict = {}
                                            controller.ica_dict = {}
                                            snr_score = torch.as_tensor(metrics["snr"], device=device_ref, dtype=torch.float32)
                                            timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}

                                # ================ ## OPTIMIZATION     ## ================ #
                                for cfg_name in config_dictionary:

                                    config_dict = config_dictionary[cfg_name]

                                    if cfg_name == 'case1':

                                        pooled_prompt_embeds = pooled_prompt_embeds.to(device_ref).to(dtype_ref)
                                        base_latents = base_latents.to(device_ref, non_blocking=True)
                                        src_latents = src_latents.to(device_ref, non_blocking=True)
                                        image_latents_4d = image_latents_4d.to(device_ref, non_blocking=True)
                                        prev_pair = None
                                        alpha_params = config_dict
                                        for epoch in range(max_epochs):

                                            set_alphas_for_case(pipe.transformer, 'Kontext', controller, cfg_name, alpha_params)
                                            self.sd.pipeline.transformer.train()
                                            _, y0_pil, metrics, noise_pred = self.sd.pipeline.onetime_inference(
                                                latents=lat_map_cpu[cfg_name].to(device, non_blocking=True).to('cuda'),
                                                image_latents=src_latents.to(device),
                                                image_latents_4d=image_latents_4d.to(device),
                                                t=t,
                                                i=stage_idx,
                                                guidance=guidance,
                                                start_noise=base_latents.to(device_ref, non_blocking=True),
                                                pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                                                prompt_embeds=prompt_embeds.to(device),
                                                text_ids=text_ids,
                                                latent_ids=latent_ids,
                                                timesteps=timesteps)

                                            y0_pil.save(os.path.join(unique_folder,f"{cfg_name}_{stage_idx}_{epoch}.png"))
                                            noise_pred_dict[cfg_name] = noise_pred.detach()
                                            embedding_dictionary = controller.embedding_dictionary
                                            casewise_embedding_dict[cfg_name] = embedding_dictionary
                                            controller.embedding_dictionary = {}

                                            tcas = 0
                                            icas = 0
                                            tca_dict = controller.tca_dict
                                            ica_dict = controller.ica_dict
                                            for layer in tca_dict.keys():
                                                tcas += tca_dict[layer]
                                                icas += ica_dict[layer]
                                            tca_ratio = tcas / (tcas + icas)
                                            tca1 = tca_ratio
                                            controller.reset()
                                            controller.tca_dict = {}
                                            controller.ica_dict = {}
                                            snr1 = torch.as_tensor(metrics["snr"], device=device_ref, dtype=torch.float32)
                                            timewise_metrics[cfg_name] = {"snr": snr1, 'tca_ratio': tca_ratio}

                                            with open(logging_dir, "a") as ftxt:
                                                ftxt.write(f'{stage_idx},{epoch},{snr_score},{tca_ratio}\n')
                                            if epoch == 0:
                                                snr2 = timewise_metrics['case2']['snr'].to(device_ref, dtype=torch.float32).detach()
                                                snr3 = timewise_metrics['case3']['snr'].to(device_ref, dtype=torch.float32).detach()
                                                # ===================================================================
                                                # Switching Values .............
                                                # ===================================================================
                                                if snr2 < snr3 :
                                                    before3 = config_dictionary['case3']
                                                    before2 = config_dictionary['case2']
                                                    config_dictionary['case3'] = before2
                                                    config_dictionary['case2'] = before3
                                                    before_metric2 = timewise_metrics['case2']
                                                    before_metric3 = timewise_metrics['case3']
                                                    timewise_metrics['case2'] = before_metric3
                                                    timewise_metrics['case3'] = before_metric2
                                                    before_emb2 = casewise_embedding_dict['case2']
                                                    before_emb3 = casewise_embedding_dict['case3']
                                                    casewise_embedding_dict['case2'] = before_emb3
                                                    casewise_embedding_dict['case3'] = before_emb2
                                                    snr2 = timewise_metrics['case2']['snr'].to(device_ref,
                                                                                               dtype=torch.float32).detach()
                                                    snr3 = timewise_metrics['case3']['snr'].to(device_ref,
                                                                                               dtype=torch.float32).detach()

                                                target_snr = (snr2 + snr3) / 2.0
                                                target_diff1 = abs(snr1 - target_snr) # more big (want to control snr)
                                                tca2 = timewise_metrics['case2']['tca_ratio'].to(device_ref,dtype=torch.float32).detach()
                                                tca3 = timewise_metrics['case3']['tca_ratio'].to(device_ref,dtype=torch.float32).detach()
                                                target_tca = (tca2 + tca3) / 2.0
                                                target_diff2 = abs(tca1 - target_tca)
                                                if target_diff1 > target_diff2:
                                                    target = target_snr
                                                    diff_2 = torch.abs(snr2-snr1)
                                                    diff_3 = torch.abs(snr1-snr3)
                                                    if diff_2 > diff_3 :
                                                        following = 'case2'
                                                    else :
                                                        following = 'case3'
                                                    reference_embedding_dict = casewise_embedding_dict[following]
                                                    criteria = abs(target - snr1)
                                                    check_snr = True
                                                else:
                                                    target = target_tca
                                                    diff_2 = torch.abs(tca2 - tca_ratio)
                                                    diff_3 = torch.abs(tca3 - tca_ratio)
                                                    if diff_2 > diff_3 :
                                                        following = 'case2'
                                                    else :
                                                        following = 'case3'
                                                    reference_embedding_dict = casewise_embedding_dict[following]
                                                    criteria = abs(target - tca1)
                                                    check_snr = False

                                                print(f' following + {following}')
                                                def build_alpha_params(config_dict,
                                                                       train_names, device=None,
                                                                       dtype=torch.float32):
                                                    alpha_params = nn.ParameterDict()
                                                    for layer_name, alpha_init in config_dict.items():
                                                        requires = layer_name in train_names
                                                        alpha_params[layer_name] = nn.Parameter(torch.tensor(float(alpha_init), device=device, dtype=dtype),requires_grad=requires)
                                                    return alpha_params

                                                if following == 'case2':
                                                    # training only case2
                                                    train_set = case2_trainings
                                                else :
                                                    train_set =  case3_trainings
                                                train_set = case2_trainings + case3_trainings
                                                alpha_params = build_alpha_params(config_dict,
                                                                                  train_set,
                                                                                  device=device_ref,
                                                                                  dtype=torch.float32)
                                                opt = Adam([{"params": alpha_params.parameters(), "lr": 1.0, "weight_decay": 0.0}], betas=(0.9, 0.999), eps=30)
                                                opt = accelerator.prepare(opt)

                                            if epoch > 0 :
                                                w_snr = 1.0
                                                w_tca = 1.0
                                                if check_snr :
                                                    criteria_t = abs(target - snr1)
                                                else :
                                                    criteria_t = abs(target - tca1)
                                                loss_snr = F.mse_loss(target_snr, snr1)
                                                loss_tca = F.mse_loss(target_tca, tca1)
                                                loss_main = w_snr * loss_snr + w_tca * loss_tca
                                                loss_reg = 0
                                                for layer_name in reference_embedding_dict:
                                                    txt_ref, img_ref = reference_embedding_dict[layer_name]['txt'], reference_embedding_dict[layer_name]['img']
                                                    train_ref, train_img = embedding_dictionary[layer_name]['txt'], embedding_dictionary[layer_name]['img']
                                                    loss_reg += F.mse_loss(txt_ref, train_ref)
                                                    loss_reg += F.mse_loss(img_ref, train_img)
                                                w_main = self.train_config.w_main
                                                w_reg = self.train_config.w_reg
                                                total_loss = w_main * loss_main + w_reg * loss_reg
                                                l1 = float(loss_main.detach().cpu())
                                                l2 = float(loss_reg.detach().cpu())
                                                RTOL, ATOL = 1e-6, 1e-8
                                                def _close(a, b):
                                                    return abs(a - b) <= (ATOL + RTOL * max(abs(a), abs(b)))
                                                if prev_pair is not None:
                                                    same_now = _close(l1, prev_pair[0]) and _close(l2, prev_pair[1])
                                                    if same_now :
                                                        same_pair_streak += 1
                                                    else :
                                                        same_pair_streak = 0
                                                    if same_pair_streak > 4 :
                                                        break
                                                else:
                                                    prev_pair = (l1, l2)
                                                    same_pair_streak = 0

                                                # 3번 연속 동일하면 루프 중단
                                                if same_pair_streak >= 3:
                                                    print(f"[STOP] loss pair repeated 3x: loss1={l1:.6f}, reg_loss={l2:.6f}")
                                                    break
                                                print(f' [{t}] {stage_idx}/{num_steps}  snr1 {snr1} snr2 {snr2} snr3 {snr3} tca1 {tca1} tca2 {tca2} tca3 {tca3} '
                                                      f' loss_snr {loss_snr} loss_tca {loss_tca} loss_reg {loss_reg} total_loss {total_loss}')
                                                total_loss = total_loss.float()
                                                opt.zero_grad(set_to_none=True)
                                                accelerator.backward(total_loss)
                                                del total_loss, snr1,
                                                opt.step()
                                                flush()
                                                if stage_idx < stage_1 :
                                                    if criteria_t < criteria * 0.3 :
                                                        break
                                                else :
                                                    snr1, snr2, snr3 = timewise_metrics['case1']['snr'],       timewise_metrics['case2']['snr']      ,timewise_metrics['case3']['snr']
                                                    tca1, tca2, tca3 = timewise_metrics['case1']['tca_ratio'], timewise_metrics['case2']['tca_ratio'],timewise_metrics['case3']['tca_ratio']
                                                    if snr2 > snr1 and snr1 > snr3 and tca1 > tca2 and tca3 > tca1 :
                                                        break
                                config_dictionary['case1'] = alpha_params
                                if max_epochs > 0 :
                                    del opt, noise_pred, embedding_dictionary,casewise_embedding_dict, timewise_metrics, snr2, snr3, target_snr, target_tca
                                return noise_pred_dict, lat_map_cpu


                            if stage_idx  < stage_1 :
                                noise_pred_dict, lat_map_cpu = do_stage1(lat_map_cpu, t, src_latents, image_latents_4d, pooled_prompt_embeds, base_latents)
                            else :
                                controller.stage = 'stage2'
                                torch.cuda.reset_peak_memory_stats()
                                torch.backends.cudnn.benchmark = True
                                casewise_embedding_dict = {}
                                timewise_metrics = {}
                                noise_pred_dict = {}
                                for cfg_name in config_dictionary :
                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name == 'case1':
                                        pooled_prompt_embeds = pooled_prompt_embeds.to(device_ref).to(dtype_ref)
                                        base_latents = base_latents.to(device_ref, non_blocking=True)
                                        src_latents = src_latents.to(device_ref, non_blocking=True)
                                        image_latents_4d = image_latents_4d.to(device_ref, non_blocking=True)
                                        set_alphas_for_case(pipe.transformer, 'Kontext', controller, cfg_name,config_dict)
                                        with torch.no_grad():
                                            _, y0_pil, metrics, noise_pred = self.sd.pipeline.onetime_inference(
                                                latents=lat_map_cpu[cfg_name].to(device, non_blocking=True).to('cuda'),
                                                image_latents=src_latents.to(device),
                                                image_latents_4d=image_latents_4d.to(device),
                                                t=t,
                                                i=stage_idx,
                                                guidance=guidance,
                                                start_noise=base_latents.to(device_ref, non_blocking=True),
                                                pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                                                prompt_embeds=prompt_embeds.to(device),
                                                text_ids=text_ids,
                                                latent_ids=latent_ids,
                                                timesteps=timesteps)
                                        y0_pil.save(os.path.join(unique_folder, f"{cfg_name}_{stage_idx}.png"))
                                        noise_pred_dict[cfg_name] = noise_pred.detach()
                                        embedding_dictionary = controller.embedding_dictionary
                                        casewise_embedding_dict[cfg_name] = embedding_dictionary
                                        controller.embedding_dictionary = {}
                                        tcas = 0
                                        icas = 0
                                        tca_dict = controller.tca_dict
                                        ica_dict = controller.ica_dict
                                        for layer in tca_dict.keys():
                                            tcas += tca_dict[layer]
                                            icas += ica_dict[layer]
                                        tca_ratio = tcas / (tcas + icas)
                                        controller.reset()
                                        controller.tca_dict = {}
                                        controller.ica_dict = {}
                                        snr1 = torch.as_tensor(metrics["snr"], device=device_ref, dtype=torch.float32)
                                        timewise_metrics[cfg_name] = {"snr": snr1, 'tca_ratio': tca_ratio}
                                        with open(logging_dir, "a") as ftxt:
                                            ftxt.write(f'{stage_idx},-,{snr1},{tca_ratio}\n')
                                        if search :
                                            embedding_dictionary = controller.embedding_dictionary
                                            controller.embedding_dictionary = {}
                                        else :
                                            controller.embedding_dictionary = {}


                                    if cfg_name != 'case1':
                                        if search:
                                            with torch.no_grad():
                                                set_alphas_for_case(pipe.transformer,
                                                                    'Kontext',
                                                                    controller,
                                                                    cfg_name,
                                                                    config_dict)
                                                _, y0_pil, metrics, noise_pred = self.sd.pipeline.onetime_inference(
                                                    latents=lat_map_cpu[cfg_name].to(device, non_blocking=True).to('cuda'),
                                                    image_latents=src_latents.to(device),
                                                    image_latents_4d=image_latents_4d.to(device),
                                                    t=t,
                                                    i=stage_idx,
                                                    guidance=guidance,
                                                    start_noise=base_latents.to(device_ref, non_blocking=True),
                                                    pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                                                    prompt_embeds=prompt_embeds.to(device),
                                                    text_ids=text_ids,
                                                    latent_ids=latent_ids,
                                                    timesteps=timesteps)
                                                y0_pil.save(os.path.join(unique_folder, f"{cfg_name}_{stage_idx}.png"))
                                                noise_pred_dict[cfg_name] = noise_pred.detach()
                                                embedding_dictionary = controller.embedding_dictionary
                                                casewise_embedding_dict[cfg_name] = embedding_dictionary
                                                controller.embedding_dictionary = {}

                                                tcas = 0
                                                icas = 0
                                                tca_dict = controller.tca_dict
                                                ica_dict = controller.ica_dict
                                                for layer in tca_dict.keys():
                                                    tcas += tca_dict[layer]
                                                    icas += ica_dict[layer]
                                                tca_ratio = tcas / (tcas + icas)
                                                controller.reset()
                                                controller.tca_dict = {}
                                                controller.ica_dict = {}
                                                snr_score = torch.as_tensor(metrics["snr"], device=device_ref,
                                                                            dtype=torch.float32)
                                                timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}
                                # ===================================
                                # Stage 2
                                # ===================================
                                if search:

                                    snr2 = timewise_metrics['case2']['snr']
                                    snr3 = timewise_metrics['case3']['snr']
                                    snr1 = timewise_metrics['case1']['snr']
                                    tca2 = timewise_metrics['case2']['tca_ratio']
                                    tca3 = timewise_metrics['case3']['tca_ratio']
                                    tca1 = timewise_metrics['case1']['tca_ratio']

                                    if snr2 > snr1:
                                        decision_snr = True
                                        if snr1 > snr3:
                                            decision_snr = True
                                        else :
                                            decision_snr = False
                                    else :
                                        decision_snr = False

                                    if tca2 < tca1:
                                        decision_tca = True
                                        if tca1 <= tca3:
                                            decision_tca = True
                                        else :
                                            decision_tca = False
                                    else :
                                        decision_tca = False
                                    print(f' decision_snr : {decision_snr} and target_tca {decision_tca} ')

                                    print(f' [{t}] {stage_idx}/{num_steps}  snr1 {snr_score} snr2 {snr2} snr3 {snr3} tca1 {tca_ratio} tca2 {tca2} tca3 {tca3} ')
                                    def set_midium(org_dict, ref_dict):
                                        for layer in org_dict.keys():
                                            org_dict[layer] = (org_dict[layer] + ref_dict[layer]) / 2
                                        return org_dict

                                    if decision_snr  :
                                        if not decision_tca:
                                            ref_dict = config_dictionary['case3']
                                            config_dictionary['case1'] = set_midium(config_dictionary['case1'], ref_dict)
                                            print(f' [Stage2] Update CASE1 config dictionary on {t}')
                                            del snr2, snr3, tca2, tca3, decision_tca, embedding_dictionary
                                        else :
                                            result_check.append(1)

                                            def to_bool(x):
                                                if isinstance(x, torch.Tensor):
                                                    # 스칼라 텐서 가정; 필요시 .any() 등으로 확장 가능
                                                    return bool(x.detach().item())
                                                return bool(x)

                                            result_window = deque(maxlen=3)
                                            ok = to_bool(decision_tca) and to_bool(decision_snr)
                                            result_window.append(1 if ok else 0)
                                            print(f' [Stage2] Preserve CASE1 config dictionary on {t}')

                                            if len(result_window) == 3 and sum(result_window) == 3:
                                                search = False
                                                del config_dictionary["case2"]
                                                del config_dictionary["case3"]
                                                del snr2, snr3, tca2, tca3, decision_tca, embedding_dictionary

                                    else :

                                        if not decision_tca:
                                            print(
                                                f' [Stage2] Bad case go to Stage1 on {t} with snr1 {snr_score} snr2 {snr2}  snr3 {snr3} and tca1 {tca_ratio} tca2 {tca2} tca3 {tca3}')
                                            del snr2, snr3, tca2, tca3, decision_tca, embedding_dictionary
                                            noise_pred_dict, lat_map_cpu = do_stage1(lat_map_cpu, t, src_latents,
                                                                                     image_latents_4d,
                                                                                     pooled_prompt_embeds, base_latents)

                                        else :
                                            ref_dict = config_dictionary['case2']
                                            config_dictionary['case1'] = set_midium(config_dictionary['case1'], ref_dict)
                                            print(f' [Stage2] Update CASE1 config dictionary on {t}')
                                            del snr2, snr3, tca2, tca3, decision_tca, embedding_dictionary


                            # =============================================================================
                            # latent update                                                               #
                            # =============================================================================
                            sigma_memory = []
                            for config_idx, cfg_name in enumerate(noise_pred_dict.keys()):
                                noise_pred = noise_pred_dict[cfg_name].detach()
                                noise_pred = noise_pred.to('cuda')
                                if config_idx == 0:
                                    sigma, sigma_next = None, None
                                else:
                                    sigma, sigma_next = sigma_memory[0], sigma_memory[1]
                                latent_next, sigma, sigma_next = self.sd.pipeline.scheduler.step(noise_pred.to(device),
                                                                                                 t,
                                                                                                 lat_map_cpu[cfg_name].to(device),
                                                                                                 sigma=sigma,
                                                                                                 sigma_next=sigma_next,
                                                                                                 return_dict=False)
                                sigma_memory.append(sigma)
                                sigma_memory.append(sigma_next)
                                lat_map_cpu[cfg_name] = latent_next

                    end_time = time.time()
                    with open(record_dir, 'a')  as f :
                        f.write(f'time, {end_time-start_time}\n')
                    print(" step 8. decode to image")
                    for case_key, lat_cpu_final in lat_map_cpu.items():
                        if case_key == 'case1':
                            lat_gpu_final = lat_cpu_final.to(device, non_blocking=True)
                            latents = pipe._unpack_latents(lat_gpu_final, 1024, 1024, pipe.vae_scale_factor)
                            latents = ((latents / pipe.vae.config.scaling_factor)
                                    + pipe.vae.config.shift_factor)
                            decoded = pipe.vae.decode(latents, return_dict=False)[0]
                            pil_img = pipe.image_processor.postprocess(
                                decoded, output_type="pil"
                            )[0]
                            save_path = os.path.join(
                                result_folder,
                                f"{out_name}.png")
                            pil_img.save(save_path)

class ExtensionJob():
    def __init__(self, config_total: OrderedDict):
        def get_conf(config, key, default=None, required=False):
            if key in config:
                return config[key]
            else:
                return default
        main_config = config_total['config']
        self.raw_config = config_total
        self.job = config_total['job']
        self.name = get_conf(main_config, 'name', required=True)
        if 'meta' in config_total:
            self.meta = config_total['meta']
        else:
            self.meta = OrderedDict()
        self.device = get_conf(main_config, 'device', 'cpu')
        self.process_dict = get_all_extensions_process_dict()
        config = main_config['process'][0]
        self.config = config
    def set_process(self):
        self.process = SDTrainer(0, self, self.config)

    def run(self, args):
        self.process.run(args)

def main(args):

    print(f' 1. logger')
    if args.log is not None:
        setup_log_to_file(args.log)
    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    print_acc(f" 2. Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")
    config_file = config_file_list[0]
    config_total = get_config(config_file)
    job = ExtensionJob(config_total)
    job.set_process()
    job.run(args)


if __name__ == '__main__':
    print(f' HELLOW')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )
    parser.add_argument(
        '-r', '--target', type=int
    )
    parser.add_argument('--add_idx', type = int)
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    parser.add_argument('--max_epochs',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    parser.add_argument('--max_epochs_add',
        type=int,
        default=0,)
    parser.add_argument('--stage_1_add',
                        type=int,
                        default=0, )
    #max_epochs = d.max_epochs + args.
    #stage_1 = d.stage_1 + args.stage_1_add

    args = parser.parse_args()
    main(args)