import os 
import torch
from abc import *
from pathlib import Path
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline

from diffusion.stable_diffusion import StableDiffusion


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

    
class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.init_model()
        self.init_mapper()
        
    def initialize(self):
        now = get_current_time()
        save_top_dir = self.config.save_top_dir
        tag = self.config.tag
        save_dir_now = self.config.save_dir_now 
        
        if save_dir_now:
            self.output_dir = Path(save_top_dir) / f"{tag}/{now}"
        else:
            self.output_dir = Path(save_top_dir) / f"{tag}"
        
        if not os.path.isdir(self.output_dir):
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            print(f"Results exist in the output directory, use time string to avoid name collision.")
            exit(0)
            
        print("[*] Saving at ", self.output_dir)
    
    
    @abstractmethod
    def init_mapper(self, **kwargs):
        pass
    
    
    @abstractmethod
    def forward_mapping(self, z_t, **kwargs):
        pass
    
    
    @abstractmethod
    def inverse_mapping(self, x_ts, **kwargs):
        pass
    
    
    @abstractmethod
    def compute_noise_preds(self, xts, ts, **kwargs):
        pass
        
    
    def init_model(self):
        if self.config.model == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(self.device)
            
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            self.model = StableDiffusion(
                **pipe.components,
            )
            
            del pipe
            
        elif self.config.model == "deepfloyd":

            self.stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            )
            self.stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            
            scheduler = DDIMScheduler.from_config(self.stage_1.scheduler.config)
            self.stage_1.scheduler = self.stage_2.scheduler = scheduler
            
        else:
            raise NotImplementedError(f"Invalid model: {self.config.model}")
        
        
        if self.config.model in ["sd"]:
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            if hasattr(self.model, "vae"):
                self.model.vae.requires_grad_(False)
        else:
            self.stage_1.text_encoder.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            
            self.stage_1 = self.stage_1.to(self.device)
            self.stage_2 = self.stage_2.to(self.device)
                
                
    def compute_tweedie(self, xts, eps, timestep, alphas, sigmas, **kwargs):
        """
        Input:
            xts, eps: [B,*]
            timestep: [B]
            x_t = alpha_t * x0 + sigma_t * eps
        Output:
            pred_x0s: [B,*]
        """

        # print(xts.shape, eps.shape)
        if eps.shape[1] > xts.shape[1]:
            eps = eps[:, :xts.shape[1], :, :]
        
        
        pred_x0s = (1 / (alphas[timestep])) * (xts - (sigmas[timestep]) * eps)


        return pred_x0s

        
    def compute_prev_state(
        self, xts, pred_x0s, timestep, **kwargs,
    ):
        """
        Input:
            xts: [N,C,H,W]
        Output:
            pred_prev_sample: [N,C,H,W]
        """

        if self.config.app == "ambiguous_image":
            scheduler = self.stage_1.scheduler
        else:
            scheduler = self.model.scheduler

        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        

        t = timestep
        prev_t = prev_timestep


        alphas_cumprod = scheduler.alphas_cumprod
        alphas = scheduler.alphas
        betas = scheduler.betas

        if prev_t >= 0:
            alpha_t_1 = alphas[prev_t]
            alphas_cumprod_t_1 = alphas_cumprod[prev_t]
        else:
            alpha_t_1 = alphas[1]
            alphas_cumprod_t_1 = alphas_cumprod[0]

        alphas_cumprod_t = alphas_cumprod[t]
        
        alpha_t = alphas[t]
        
        beta_t = betas[t]
        

        # pred_prev_sample = ((alpha_t ** 0.5) * (1 - alphas_cumprod_t_1))/(1 - alphas_cumprod_t) * xts + \
        #                     (((alphas_cumprod_t_1 ** 0.5) * beta_t) / (1 - alphas_cumprod_t)) * pred_x0s

        pred_prev_sample = (alphas_cumprod_t_1 ** (0.5)) * pred_x0s + \
                            (((1 - alphas_cumprod_t_1)/(1 - alphas_cumprod_t)) ** (0.5)) * (xts - (alphas_cumprod_t ** (0.5)) * pred_x0s)

        # pred_prev_sample = torch.sqrt(alphas_cumprod_t_1) * pred_x0s + torch.sqrt((1 - alphas_cumprod_t_1)/(1-alphas_cumprod_t))*(xts-torch.sqrt(alphas_cumprod_t)*pred_x0s)
        
        # pred_prev_sample = (alphas_cumprod_t_1 ** 0.5) * pred_x0s + ((1 - alphas_cumprod_t_1) ** 0.5) * xts 
        
        # TODO: Implement compute_prev_state
        # raise NotImplementedError("compute_prev_state is not implemented yet.")
        return pred_prev_sample
        
    def one_step_process(
        self, input_params, timestep, alphas, sigmas, case,  **kwargs
    ):
        """
        Input:
            latents: either xt or zt. [B,*]
        Output:
            output: the same with latent.
        """
        
        
        
        # Synchronization using SyncTweedies 
        # --------------------------------

        if case == 1:
            
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            if eps_preds.shape[1] > xts.shape[1]:
                eps_preds = eps_preds[:, :xts.shape[1], :, :]

            # eps_preds_big = self.inverse_mapping(eps_preds, var_type="tweedie", **kwargs)
            # eps_preds = self.forward_mapping(eps_preds_big, bg=eps_preds, **kwargs)
            
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )

            # eps_preds_big = self.inverse_mapping(eps_preds, var_type="tweedie", **kwargs)
            # eps_preds = self.forward_mapping(eps_preds_big, bg=eps_preds, **kwargs)

            # z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
            # x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

            z_t_1 = self.inverse_mapping(x_t_1, var_type="tweedie", **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, bg=x_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

        # --------------------------------
        elif case == 2:
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            if eps_preds.shape[1] > xts.shape[1]:
                eps_preds = eps_preds[:, :xts.shape[1], :, :]

            eps_preds_big = self.inverse_mapping(eps_preds, var_type="tweedie", **kwargs)
            eps_preds = self.forward_mapping(eps_preds_big, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

        # --------------------------------
        elif case == 3:
            
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            if eps_preds.shape[1] > xts.shape[1]:
                eps_preds = eps_preds[:, :xts.shape[1], :, :]

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )

            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization
            
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
        # ----------------------------------
        elif case == 4:
            xts = input_params["xts"]

            zts = self.inverse_mapping(xts, var_type="tweedie", **kwargs)
            xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            if eps_preds.shape[1] > xts.shape[1]:
                eps_preds = eps_preds[:, :xts.shape[1], :, :]

            # eps_preds_big = self.inverse_mapping(eps_preds, var_type="tweedie", **kwargs)
            # eps_preds = self.forward_mapping(eps_preds_big, bg=eps_preds, **kwargs)
            
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )

            # eps_preds_big = self.inverse_mapping(eps_preds, var_type="tweedie", **kwargs)
            # eps_preds = self.forward_mapping(eps_preds_big, bg=eps_preds, **kwargs)

            # z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
            # x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

            # z_t_1 = self.inverse_mapping(x_t_1, var_type="tweedie", **kwargs)
            # x_t_1 = self.forward_mapping(z_t_1, bg=x_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

        return out_params