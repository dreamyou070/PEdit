import os
import sys
from base_utils.scheduling import FlowMatchEulerDiscreteScheduler as CustomFMEDS
from base_utils.custom_editing_inference import onetime_inference_qwen
from types import MethodType
import argparse
from collections import deque
import torch
import torch.nn.functional as F
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

from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file
from toolkit.extension import get_all_extensions_process_dict
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import (
    add_all_snr_to_noise_scheduler,
)
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
from toolkit.unloader import unload_text_encoder
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
# from lycoris.config import PRESET
from torch.utils.data import DataLoader
import torch
import torch.backends.cuda
try:
    from huggingface_hub import HfApi
except Exception:
    raise
from toolkit.basic import value_map
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.ema import ExponentialMovingAverage
from toolkit.embedding import Embedding
from toolkit.image_utils import show_tensors, show_latents, reduce_contrast
from toolkit.ip_adapter import IPAdapter
from toolkit.models.decorator import Decorator
from toolkit.network_mixins import Network
from base_utils.base import save_config, load_image_bhwc_uint8
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit.saving import save_t2i_from_diffusers, load_t2i_model, save_ip_adapter_from_diffusers, \
    load_ip_adapter_model, load_custom_adapter_model
from toolkit.config import get_config
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.stable_diffusion_model import StableDiffusion
from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_base_model_info_to_meta, \
    parse_metadata_from_safetensors
from toolkit.train_tools import get_torch_dtype, LearnableSNRGamma, apply_learnable_snr_gos, apply_snr_weight
import gc
from tqdm import tqdm

from toolkit.config_modules import SaveConfig, LoggingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig, \
    GenerateImageConfig, EmbeddingConfig, DatasetConfig, preprocess_dataset_raw_config, AdapterConfig, GuidanceConfig, \
    validate_configs, \
    DecoratorConfig
from toolkit.logging_aitk import create_logger
from diffusers import FluxTransformer2DModel
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.print import print_acc
from accelerate import Accelerator
import transformers
import diffusers
import hashlib

from toolkit.util.blended_blur_noise import get_blended_blur_noise
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
        #self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        #first_sample_config = self.get_conf('first_sample', None)
        #if first_sample_config is not None:
        #    self.has_first_sample_requested = True
        #    self.first_sample_config = SampleConfig(**first_sample_config)
        #else:
        #    self.has_first_sample_requested = False
        #    self.first_sample_config = self.sample_config
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
        #
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
        trg_file_idx = args.target
        for file_idx, filename in enumerate(os.listdir(src_folder)):

            if file_idx == trg_file_idx :
                print(f' load model ... ')
                BaseTrainProcess.run(self)
                self.hook_before_model_load()
                model_config_to_load = copy.deepcopy(self.model_config)

                ModelClass = get_model_class(self.model_config)
                if hasattr(ModelClass, 'get_train_scheduler'):
                    sampler = ModelClass.get_train_scheduler()
                else:
                    arch = 'sd'
                    if self.model_config.is_pixart:
                        arch = 'pixart'
                    if self.model_config.is_flux:
                        arch = 'flux'
                    if self.model_config.is_lumina2:
                        arch = 'lumina2'
                    sampler = get_sampler(
                        self.train_config.noise_scheduler,
                        {
                            "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
                        },
                        arch=arch,
                    )

                if self.train_config.train_refiner and self.model_config.refiner_name_or_path is not None and self.network_config is None:
                    previous_refiner_save = self.get_latest_save_path(self.job.name + '_refiner')
                    if previous_refiner_save is not None:
                        model_config_to_load.refiner_name_or_path = previous_refiner_save
                        self.load_training_state_from_metadata(previous_refiner_save)

                self.sd = ModelClass(
                    # todo handle single gpu and multi gpu here
                    # device=self.device,
                    device=self.accelerator.device,
                    model_config=model_config_to_load,
                    dtype=self.train_config.dtype,
                    custom_pipeline=self.custom_pipeline,
                    noise_scheduler=sampler,
                )

                self.hook_after_sd_init_before_load()
                self.sd.load_model()

                if self.model_config.compile:
                    try:
                        torch.compile(self.sd.unet, dynamic=True, fullgraph=True, mode='max-autotune')
                    except Exception as e:
                        print_acc(f"Failed to compile model: {e}")
                        print_acc("Continuing without compilation")

                dtype = get_torch_dtype(self.train_config.dtype)

                # model is loaded from BaseSDProcess
                unet = self.sd.unet
                vae = self.sd.vae
                tokenizer = self.sd.tokenizer
                text_encoder = self.sd.text_encoder
                noise_scheduler = self.sd.noise_scheduler

                if self.train_config.xformers:
                    vae.enable_xformers_memory_efficient_attention()
                    unet.enable_xformers_memory_efficient_attention()
                    if isinstance(text_encoder, list):
                        for te in text_encoder:
                            # if it has it
                            if hasattr(te, 'enable_xformers_memory_efficient_attention'):
                                te.enable_xformers_memory_efficient_attention()

                if self.train_config.attention_backend != 'native':
                    if hasattr(vae, 'set_attention_backend'):
                        vae.set_attention_backend(self.train_config.attention_backend)
                    if hasattr(unet, 'set_attention_backend'):
                        unet.set_attention_backend(self.train_config.attention_backend)
                    if isinstance(text_encoder, list):
                        for te in text_encoder:
                            if hasattr(te, 'set_attention_backend'):
                                te.set_attention_backend(self.train_config.attention_backend)
                    else:
                        if hasattr(text_encoder, 'set_attention_backend'):
                            text_encoder.set_attention_backend(self.train_config.attention_backend)
                if self.train_config.sdp:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)

                if self.train_config.gradient_checkpointing:
                    # if has method enable_gradient_checkpointing
                    if hasattr(unet, 'enable_gradient_checkpointing'):
                        unet.enable_gradient_checkpointing()
                    elif hasattr(unet, 'gradient_checkpointing'):
                        unet.gradient_checkpointing = True
                    else:
                        print("Gradient checkpointing not supported on this model")
                    if isinstance(text_encoder, list):
                        for te in text_encoder:
                            if hasattr(te, 'enable_gradient_checkpointing'):
                                te.enable_gradient_checkpointing()
                            if hasattr(te, "gradient_checkpointing_enable"):
                                te.gradient_checkpointing_enable()
                    else:
                        if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                            text_encoder.enable_gradient_checkpointing()
                        if hasattr(text_encoder, "gradient_checkpointing_enable"):
                            text_encoder.gradient_checkpointing_enable()

                if self.sd.refiner_unet is not None:
                    self.sd.refiner_unet.to(self.device_torch, dtype=dtype)
                    self.sd.refiner_unet.requires_grad_(False)
                    self.sd.refiner_unet.eval()
                    if self.train_config.xformers:
                        self.sd.refiner_unet.enable_xformers_memory_efficient_attention()
                    if self.train_config.gradient_checkpointing:
                        self.sd.refiner_unet.enable_gradient_checkpointing()

                if isinstance(text_encoder, list):
                    for te in text_encoder:
                        te.requires_grad_(False)
                        te.eval()
                else:
                    text_encoder.requires_grad_(False)
                    text_encoder.eval()
                unet.to(self.device_torch, dtype=dtype)
                unet.requires_grad_(False)
                unet.eval()
                vae = vae.to(torch.device('cpu'), dtype=dtype)
                vae.requires_grad_(False)
                vae.eval()
                flush()
                self.sd.set_device_state(self.train_device_state_preset)
                flush()
                # sself.ensure_params_requires_grad(force=True)
                ###################################################################
                # Personalize
                ###################################################################
                print(" step 1. pipeline customize with selt forward and scheduling")
                self.sd.pipeline.enable_attention_slicing()
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                self.sd.pipeline.onetime_inference_qwen = MethodType(onetime_inference_qwen, self.sd.pipeline)

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

                print(f'src_path = {src_path}')

                src_img = Image.open(src_path).convert("RGB").resize((1024, 1024))
                txt_path = os.path.join(inst_folder, f"{name}.txt")
                with open(txt_path, "r") as f:
                    editing_prompt = f.readlines()[0].strip()

                print(" step 6. preparing")
                print(f' +++ name = {name}')
                unique_folder = os.path.join(result_folder, name)
                print(f'unique_folder : {unique_folder}')
                if not os.path.exists(unique_folder):
                    os.makedirs(unique_folder, exist_ok=True)
                    record_dir = os.path.join(unique_folder, 'timewise_score.txt')
                    with open(record_dir, "w") as ftxt:
                        ftxt.write(f'editing prompt : {editing_prompt}\n')
                    with open(record_dir, "a") as ftxt:
                        ftxt.write(f'cfg_name,tca_ratio,snr_score\n')
                    num_steps = 28
                    configs = [("case1", 1, 1), ("case2", 8, 0), ("case3",0,8)]
                    model_name = 'Qwen'
                    config_dictionary = {}
                    for config in configs:
                        config_name, s1, s2 = config
                        config_dictionary[config_name], training_models = set_alpha_dict(model_name,
                                                                                         self.sd.pipeline.transformer,
                                                                                         config_name, s1, s2)
                        if config_name == 'case2':
                            case2_trainings = training_models
                        if config_name == 'case3':
                            case3_trainings = training_models
                    print(" step 7. config setup")
                    device_ref = 'cuda'
                    base_latents, src_latents, guidance, prompt_embeds_mask, prompt_embeds, img_shapes, txt_seq_lens, timesteps, image_latents_4d = self.sd.pipeline.prepare_inference(
                        num_inference_steps=num_steps,
                        prompt=editing_prompt,
                        image=src_img)
                    controller.valid_len = txt_seq_lens[0]
                    lat_map_cpu = {cfg_name: base_latents.detach().clone().to("cpu") for cfg_name in config_dictionary}
                    pipe = self.sd.pipeline
                    controller.state = 'save'
                    device = 'cuda'
                    stage_1 = 3
                    search = True

                    print(" step 8. Optimization")
                    pipe.transformer.to('cuda')
                    model_dev = next(pipe.transformer.parameters()).device
                    model_dtype = next(pipe.transformer.parameters()).dtype  # bf16로 기대
                    to_like = lambda x: x.to(device=model_dev, dtype=model_dtype, non_blocking=True)

                    for name, param in self.sd.pipeline.transformer.named_parameters():
                        param.to('cuda')
                    with pipe.progress_bar(total=num_steps) as progress_bar:

                        for stage_idx, t in enumerate(timesteps):
                            result_check = []

                            # ===============================================================================================================================================
                            def do_state1(lat_map_cpu, t, src_latents, image_latents_4d, base_latents):
                                controller.stage = 'stage1'
                                controller.state = 'train'
                                casewise_embedding_dict = {}
                                timewise_metrics = {}
                                noise_pred_dict = {}
                                for cfg_name in config_dictionary:

                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name != 'case1':
                                        with torch.no_grad():
                                            set_alphas_for_case(pipe.transformer,
                                                                model_name,
                                                                controller,
                                                                cfg_name,
                                                                config_dict)
                                            _, noise_pred, metrics, timewise_pil = pipe.onetime_inference_qwen(
                                                latents=to_like(lat_map_cpu[cfg_name]),
                                                image_latents=to_like(src_latents),
                                                image_latents_4d=to_like(image_latents_4d),
                                                start_noise=to_like(base_latents),
                                                t=t.to(model_dev).to(model_dtype),
                                                i=stage_idx,
                                                case_name=cfg_name,
                                                guidance=(
                                                    guidance if not torch.is_tensor(guidance) else guidance.to(model_dev,
                                                                                                               model_dtype)),
                                                encoder_hidden_states_mask=to_like(prompt_embeds_mask),
                                                encoder_hidden_states=to_like(prompt_embeds),
                                                img_shapes=img_shapes,  # 텐서가 아니면 그대로
                                                txt_seq_lens=txt_seq_lens,  # 텐서면 to_like 한 번 더
                                            )
                                            timewise_pil = self.sd.pipeline.vae.decode(timewise_pil, return_dict=False)[0][
                                                :, :, 0]
                                            timewise_pil = self.sd.pipeline.image_processor.postprocess(timewise_pil,
                                                                                                        output_type='pil')[                                                0]
                                            timewise_pil.save(                                                os.path.join(unique_folder, f"{out_name}_{cfg_name}_{stage_idx}.png"))
                                            noise_pred_dict[cfg_name] = noise_pred
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
                                                                        dtype=model_dtype)
                                            timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}
                                ##
                                for cfg_name in config_dictionary:
                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name == 'case1':
                                        max_epochs = 10
                                        prev_pair = None
                                        set_alphas_for_case(pipe.transformer,
                                                            model_name,
                                                            controller,
                                                            cfg_name,
                                                            config_dict)
                                        _, noise_pred, metrics, timewise_pil = pipe.onetime_inference_qwen(
                                            latents=to_like(lat_map_cpu[cfg_name]),
                                            image_latents=to_like(src_latents),
                                            image_latents_4d=to_like(image_latents_4d),
                                            start_noise=to_like(base_latents),
                                            t=t.to(model_dev).to(model_dtype),
                                            i=stage_idx,
                                            case_name=cfg_name,
                                            guidance=(
                                                guidance if not torch.is_tensor(guidance) else guidance.to(model_dev,
                                                                                                           model_dtype)),
                                            encoder_hidden_states_mask=to_like(prompt_embeds_mask),
                                            encoder_hidden_states=to_like(prompt_embeds),
                                            img_shapes=img_shapes,  # 텐서가 아니면 그대로
                                            txt_seq_lens=txt_seq_lens,  # 텐서면 to_like 한 번 더
                                        )
                                        noise_pred_dict[cfg_name] = noise_pred
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
                                                                    dtype=model_dtype)
                                        timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}
                                        snr2 = timewise_metrics['case2']['snr'].to(device_ref,
                                                                                   dtype=torch.bfloat16).detach()
                                        snr3 = timewise_metrics['case3']['snr'].to(device_ref,
                                                                                   dtype=torch.bfloat16).detach()
                                        target_snr = (snr2 + snr3) / 2.0
                                        target_diff1 = abs(snr_score - target_snr)  # more big (want to control snr)
                                        tca2 = timewise_metrics['case2']['tca_ratio'].to(device_ref,
                                                                                         dtype=torch.bfloat16).detach()
                                        tca3 = timewise_metrics['case3']['tca_ratio'].to(device_ref,
                                                                                         dtype=torch.bfloat16).detach()
                                        target_tca = (tca2 + tca3) / 2.0
                                        target_diff2 = abs(tca_ratio - target_tca)
                                        if target_diff1 > target_diff2:
                                            target = target_snr
                                            diff_2 = torch.abs(snr2 - snr_score)
                                            diff_3 = torch.abs(snr_score - snr3)
                                            if diff_2 > diff_3:
                                                following = 'case2'
                                            else:
                                                following = 'case3'
                                            reference_embedding_dict = casewise_embedding_dict[following]
                                            criteria = abs(target - snr_score)
                                            check_snr = True
                                        else:
                                            target = target_tca
                                            diff_2 = torch.abs(tca2 - tca_ratio)
                                            diff_3 = torch.abs(tca3 - tca_ratio)
                                            if diff_2 > diff_3:
                                                following = 'case2'
                                            else:
                                                following = 'case3'
                                            reference_embedding_dict = casewise_embedding_dict[following]
                                            criteria = abs(target - tca_ratio)
                                            check_snr = False
                                        def build_alpha_params(config_dict,
                                                               train_names, device=None,
                                                               dtype=torch.float32):
                                            alpha_params = nn.ParameterDict()
                                            for layer_name, alpha_init in config_dict.items():
                                                requires = layer_name in train_names
                                                alpha_params[layer_name] = nn.Parameter(torch.tensor(float(alpha_init), device=device, dtype=torch.bfloat16),
                                                    requires_grad=requires)
                                            return alpha_params

                                        if following == 'case2':
                                            train_set = case2_trainings
                                        else:
                                            train_set = case3_trainings
                                        train_set = case2_trainings + case3_trainings
                                        alpha_params = build_alpha_params(config_dict,
                                                                          train_set,
                                                                          device=device_ref,
                                                                          dtype=torch.bfloat16)
                                        opt = Adam(
                                            [{"params": alpha_params.parameters(), "lr": 1.0, "weight_decay": 0.0}],
                                            betas=(0.9, 0.999), eps=30)
                                        opt = accelerator.prepare(opt)

                                        for epoch in range(max_epochs):

                                            set_alphas_for_case(pipe.transformer,
                                                                model_name,
                                                                controller,
                                                                cfg_name,alpha_params)
                                            _, noise_pred, metrics, timewise_pil = pipe.onetime_inference_qwen(
                                                latents=to_like(lat_map_cpu[cfg_name]),
                                                image_latents=to_like(src_latents),
                                                image_latents_4d=to_like(image_latents_4d),
                                                start_noise=to_like(base_latents),
                                                t=t.to(model_dev).to(model_dtype),
                                                i=stage_idx,
                                                case_name=cfg_name,
                                                guidance=(
                                                    guidance if not torch.is_tensor(guidance) else guidance.to(model_dev,
                                                                                                               model_dtype)),
                                                encoder_hidden_states_mask=to_like(prompt_embeds_mask),
                                                encoder_hidden_states=to_like(prompt_embeds),
                                                img_shapes=img_shapes,  # 텐서가 아니면 그대로
                                                txt_seq_lens=txt_seq_lens,  # 텐서면 to_like 한 번 더
                                            )
                                            timewise_pil = self.sd.pipeline.vae.decode(timewise_pil.detach(), return_dict=False)[0][
                                                :, :, 0]
                                            timewise_pil = self.sd.pipeline.image_processor.postprocess(timewise_pil,
                                                                                                        output_type='pil')[
                                                0]
                                            timewise_pil.save(os.path.join(unique_folder,
                                                                           f"{out_name}_{cfg_name}_{stage_idx}_{epoch}.png"))
                                            noise_pred_dict[cfg_name] = noise_pred
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
                                            snr_score = torch.as_tensor(metrics["snr"], device=device_ref,dtype=model_dtype)
                                            timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}

                                            # ===============
                                            # LOSS
                                            # ===============
                                            w_snr = 1.0
                                            w_tca = 1.0
                                            if check_snr:
                                                criteria_t = abs(target - snr_score)
                                            else:
                                                criteria_t = abs(target - tca_ratio)
                                            loss_snr = F.mse_loss(target_snr, snr_score)
                                            loss_tca = F.mse_loss(target_tca, tca_ratio)
                                            loss_main = w_snr * loss_snr + w_tca * loss_tca
                                            loss_reg = 0
                                            for layer_name in reference_embedding_dict:
                                                txt_ref, img_ref = reference_embedding_dict[layer_name]['txt'], \
                                                    reference_embedding_dict[layer_name]['img']
                                                train_ref, train_img = embedding_dictionary[layer_name]['txt'], \
                                                    embedding_dictionary[layer_name]['img']
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
                                                if same_now:
                                                    same_pair_streak += 1
                                                else:
                                                    same_pair_streak = 0
                                                if same_pair_streak > 4:
                                                    break
                                            else:
                                                prev_pair = (l1, l2)
                                                same_pair_streak = 0
                                            if same_pair_streak >= 3:
                                                print(
                                                    f"[STOP] loss pair repeated 3x: loss1={l1:.6f}, reg_loss={l2:.6f}")
                                                break
                                            print(
                                                f' [{t}] {stage_idx}/{num_steps}  snr1 {snr_score} snr2 {snr2} snr3 {snr3} tca1 {tca_ratio} tca2 {tca2} tca3 {tca3} '
                                                f' loss_snr {loss_snr} loss_tca {loss_tca} loss_reg {loss_reg} total_loss {total_loss}')
                                            total_loss = total_loss.float()
                                            opt.zero_grad(set_to_none=True)
                                            accelerator.backward(total_loss)
                                            del total_loss, snr_score, tca_ratio
                                            opt.step()
                                            flush()
                                            if stage_idx < stage_1:
                                                if criteria_t < criteria * 0.3:
                                                    break
                                            else:
                                                snr1, snr2, snr3 = timewise_metrics['case1']['snr'], \
                                                timewise_metrics['case2']['snr'], timewise_metrics['case3']['snr']
                                                tca1, tca2, tca3 = timewise_metrics['case1']['tca_ratio'], \
                                                timewise_metrics['case2']['tca_ratio'], timewise_metrics['case3'][
                                                    'tca_ratio']
                                                if snr2 > snr1 and snr1 > snr3 and tca1 > tca2 and tca3 > tca1:
                                                        break
                                config_dictionary['case1'] = alpha_params
                                del opt, noise_pred, embedding_dictionary, casewise_embedding_dict, timewise_metrics, snr2, snr3, target_snr, target_tca
                                return noise_pred_dict, lat_map_cpu


                            if stage_idx  < stage_1 :
                                print(f' *** STAGE 1 [{stage_idx}/{timesteps}]***')
                                noise_pred_dict, lat_map_cpu = do_state1(lat_map_cpu,
                                                                         t,
                                                                         src_latents,
                                                                         image_latents_4d,
                                                                         base_latents)
                                # search = False
                            else :
                                flush()
                                torch.cuda.reset_peak_memory_stats()
                                torch.backends.cudnn.benchmark = True
                                casewise_embedding_dict = {}
                                timewise_metrics = {}
                                noise_pred_dict = {}
                                for cfg_name in config_dictionary :
                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name == 'case1':
                                        base_latents = base_latents.to(device_ref, non_blocking=True)
                                        src_latents = src_latents.to(device_ref, non_blocking=True)
                                        image_latents_4d = image_latents_4d.to(device_ref, non_blocking=True)
                                        set_alphas_for_case(pipe.transformer, model_name, controller, cfg_name,config_dict)
                                        self.sd.pipeline.transformer.train()
                                        _, noise_pred, metrics, timewise_pil = pipe.onetime_inference_qwen(
                                            latents=to_like(lat_map_cpu[cfg_name]),
                                            image_latents=to_like(src_latents),
                                            image_latents_4d=to_like(image_latents_4d),
                                            start_noise=to_like(base_latents),
                                            t=t.to(model_dev).to(model_dtype),
                                            i=stage_idx,
                                            case_name=cfg_name,
                                            guidance=(
                                                guidance if not torch.is_tensor(guidance) else guidance.to(model_dev,
                                                                                                           model_dtype)),
                                            encoder_hidden_states_mask=to_like(prompt_embeds_mask),
                                            encoder_hidden_states=to_like(prompt_embeds),
                                            img_shapes=img_shapes,  # 텐서가 아니면 그대로
                                            txt_seq_lens=txt_seq_lens,  # 텐서면 to_like 한 번 더
                                        )
                                        timewise_pil = \
                                        self.sd.pipeline.vae.decode(timewise_pil.detach(), return_dict=False)[0][
                                            :, :, 0]
                                        timewise_pil = self.sd.pipeline.image_processor.postprocess(timewise_pil,
                                                                                                    output_type='pil')[
                                            0]
                                        timewise_pil.save(os.path.join(unique_folder,
                                                                       f"{out_name}_{cfg_name}_{stage_idx}.png"))

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

                                        snr_score = torch.as_tensor(metrics["snr"], device=device_ref, dtype=model_dtype)
                                        timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}
                                        if search :
                                            embedding_dictionary = controller.embedding_dictionary
                                            controller.embedding_dictionary = {}
                                        else :
                                            controller.embedding_dictionary = {}

                                for cfg_name in config_dictionary:
                                    config_dict = config_dictionary[cfg_name]
                                    if cfg_name != 'case1':
                                        if search:
                                            with torch.no_grad():
                                                set_alphas_for_case(pipe.transformer,
                                                                    model_name,
                                                                    controller,
                                                                    cfg_name,
                                                                    config_dict)
                                                _, noise_pred, metrics, timewise_pil = pipe.onetime_inference_qwen(
                                                    latents=to_like(lat_map_cpu[cfg_name]),
                                                    image_latents=to_like(src_latents),
                                                    image_latents_4d=to_like(image_latents_4d),
                                                    start_noise=to_like(base_latents),
                                                    t=t.to(model_dev).to(model_dtype),
                                                    i=stage_idx,
                                                    case_name=cfg_name,
                                                    guidance=(
                                                        guidance if not torch.is_tensor(guidance) else guidance.to(
                                                            model_dev,
                                                            model_dtype)),
                                                    encoder_hidden_states_mask=to_like(prompt_embeds_mask),
                                                    encoder_hidden_states=to_like(prompt_embeds),
                                                    img_shapes=img_shapes,  # 텐서가 아니면 그대로
                                                    txt_seq_lens=txt_seq_lens,  # 텐서면 to_like 한 번 더
                                                )
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
                                                snr_score = torch.as_tensor(metrics["snr"], device=device_ref, dtype=model_dtype)
                                                timewise_metrics[cfg_name] = {"snr": snr_score, 'tca_ratio': tca_ratio}
                                # ===================================
                                # Stage 2
                                # ===================================
                                #search = False
                                if search:
                                    snr1 = timewise_metrics['case1']['snr'].to(device_ref, dtype=torch.float32).detach()
                                    snr2 = timewise_metrics['case2']['snr'].to(device_ref, dtype=torch.float32).detach()
                                    snr3 = timewise_metrics['case3']['snr'].to(device_ref, dtype=torch.float32).detach()
                                    target_snr = (snr2 > snr_score) and (snr_score > snr3)
                                    tca1 = timewise_metrics['case1']['tca_ratio'].to(device_ref,
                                                                                     dtype=torch.float32).detach()

                                    tca2 = timewise_metrics['case2']['tca_ratio'].to(device_ref,
                                                                                     dtype=torch.float32).detach()
                                    tca3 = timewise_metrics['case3']['tca_ratio'].to(device_ref,
                                                                                     dtype=torch.float32).detach()
                                    target_tca = (tca2 < tca_ratio) and (tca3 > tca_ratio)
                                    print(
                                        f' [{t}] {stage_idx}/{num_steps}  snr1 {snr_score} snr2 {snr2} snr3 {snr3} tca1 {tca_ratio} tca2 {tca2} tca3 {tca3} ')

                                    def set_midium(org_dict, ref_dict):
                                        for layer in org_dict.keys():
                                            org_dict[layer] = (org_dict[layer] + ref_dict[layer]) / 2
                                        return org_dict

                                    if not target_snr:
                                        ref_dict = config_dictionary['case2']
                                        print(f' [Stage2] Bad case go to Stage1 on {t} with snr1 {snr_score} snr2 {snr2}  snr3 {snr3} and tca1 {tca_ratio} tca2 {tca2} tca3 {tca3}')
                                        del snr2, snr3, tca2, tca3, target_tca, embedding_dictionary
                                        flush()
                                        noise_pred_dict, lat_map_cpu = do_state1(lat_map_cpu,
                                                                                 t,
                                                                                 src_latents,
                                                                                 image_latents_4d,
                                                                                 base_latents)

                                    else :
                                        result_check.append(1)
                                        def to_bool(x):
                                            if isinstance(x, torch.Tensor):
                                                # 스칼라 텐서 가정; 필요시 .any() 등으로 확장 가능
                                                return bool(x.detach().item())
                                            return bool(x)

                                        result_window = deque(maxlen=3)
                                        ok = to_bool(target_tca) and to_bool(target_snr)
                                        result_window.append(1 if ok else 0)
                                        print(f' [Stage2] Preserve CASE1 config dictionary on {t}')

                                        if len(result_window) == 3 and sum(result_window) == 3:
                                            search = False
                                            del config_dictionary["case2"]
                                            del config_dictionary["case3"]
                                            del snr2, snr3, tca2, tca3, embedding_dictionary

                            # =============================================================================
                            # latent update                                                               #
                            # =============================================================================
                            sigma_memory = []
                            for config_idx, cfg_name in enumerate(noise_pred_dict.keys()):
                                #if cfg_name == 'case1' :
                                noise_pred = noise_pred_dict[cfg_name].detach()
                                noise_pred = noise_pred.to('cuda')
                                if config_idx == 0:
                                    sigma, sigma_next = None, None
                                else:
                                    sigma, sigma_next = sigma_memory[0], sigma_memory[1]
                                latent_next, sigma, sigma_next = self.sd.pipeline.scheduler.step(
                                    noise_pred.to(device),
                                    t,
                                    lat_map_cpu[cfg_name].to(device),
                                    sigma=sigma,
                                    sigma_next=sigma_next,
                                    return_dict=False)
                                sigma_memory.append(sigma)
                                sigma_memory.append(sigma_next)
                                lat_map_cpu[cfg_name] = latent_next

                    print(" step 8. decode to image")
                    for case_key, lat_cpu_final in lat_map_cpu.items():
                        ref_device = 'cuda'
                        self.sd.pipeline.vae.to(ref_device)
                        latents = self.sd.pipeline._unpack_latents(lat_cpu_final.to('cuda'), 1024,1024, self.sd.pipeline.vae_scale_factor)
                        latents = latents.to(self.sd.pipeline.vae.dtype)
                        latents_mean = (
                            torch.tensor(self.sd.pipeline.vae.config.latents_mean)
                            .view(1, self.sd.pipeline.vae.config.z_dim, 1, 1, 1)
                            .to(latents.device, latents.dtype)
                        )
                        latents_std = 1.0 / torch.tensor(self.sd.pipeline.vae.config.latents_std).view(1, self.sd.pipeline.vae.config.z_dim, 1, 1,
                                                                                           1).to(
                            latents.device, latents.dtype
                        )
                        latents = latents / latents_std + latents_mean
                        image = self.sd.pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
                        pil_img = self.sd.pipeline.image_processor.postprocess(image, output_type='pil')[0]
                        save_path = os.path.join(
                            result_folder,
                            f"{out_name}.png"
                        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )
    parser.add_argument(
        '--target', type = int)
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    main(args)