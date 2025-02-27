import torch
import gc
from diffusers.utils.torch_utils import randn_tensor
import comfy.model_management as mm
from ..utils.rope_utils import get_rotary_pos_embed
from ..utils.latent_preview import prepare_callback

VAE_SCALING_FACTOR = 0.476986

class WanVideoFlowEditSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "source_embeds": ("WANVIDEMBEDS", ),
                "target_embeds": ("WANVIDEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "skip_steps": ("INT", {"default": 4, "min": 0}),
                "drift_steps": ("INT", {"default": 0, "min": 0}),
                "source_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "target_guidance_scale": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "drift_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "drift_flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 4. Allows for new frames to be generated after without looping"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, 
                model, 
                source_embeds, 
                target_embeds,
                flow_shift, 
                drift_flow_shift,
                steps, 
                skip_steps,
                drift_steps,
                source_guidance_scale, 
                target_guidance_scale,
                drift_guidance_scale,
                seed, 
                samples, 
                riflex_freq_index=0,
                force_offload=True):
        patcher = model
        model = model.model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        transformer = model["diffusion_model"]
        
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        latents = samples["samples"] * VAE_SCALING_FACTOR if samples is not None else None
        batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width = latents.shape
        height = latent_height * 8  # Assuming vae_scale_factor is 8
        width = latent_width * 8
        num_frames = (latent_num_frames - 1) * 4 + 1

        if width <= 0 or height <= 0 or num_frames <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={num_frames}"
            )
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {num_frames}"
            )

        # Get rope embeddings for the transformer
        d = transformer.dim // transformer.num_heads
        if riflex_freq_index > 0:
            from ..utils.rope_utils import rope_params
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_num_frames, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6), L_test=latent_num_frames, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6), L_test=latent_num_frames, k=riflex_freq_index)
            ], dim=1)
        else:
            freqs_cos, freqs_sin = get_rotary_pos_embed(transformer, num_frames, height, width)
            freqs_cos = freqs_cos.to(device)
            freqs_sin = freqs_sin.to(device)
        
        # Handle block swapping if configured
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                if "blocks" not in name:
                    param.data = param.data.to(device)
            
            transformer.block_swap(
                model["block_swap_args"]["blocks_to_swap"] - 1,
            )
        elif model["manual_offloading"]:
            transformer.to(device)

        mm.soft_empty_cache()
        gc.collect()
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        # Set up scheduler and timesteps
        from .wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from .wanvideo.utils.fm_solvers import get_sampling_sigmas, retrieve_timesteps
        
        # Initial scheduler with main flow shift
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            use_dynamic_shifting=False,
            algorithm_type="dpmsolver++"
        )
        sampling_sigmas = get_sampling_sigmas(steps, flow_shift)
        timesteps, _ = retrieve_timesteps(
            scheduler,
            device=device,
            sigmas=sampling_sigmas
        )
        
        # Drift scheduler with different flow shift value
        if drift_steps > 0:
            drift_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=drift_flow_shift,
                use_dynamic_shifting=False,
                algorithm_type="dpmsolver++"
            )
            drift_sampling_sigmas = get_sampling_sigmas(steps, drift_flow_shift)
            drift_timesteps, _ = retrieve_timesteps(
                drift_scheduler,
                device=device,
                sigmas=drift_sampling_sigmas
            )
            
            # Replace the later timesteps with drift timesteps
            timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        # Initialize latents
        latents = latents.to(device)

        # Setup callbacks for visualization
        from comfy.utils import ProgressBar
        from tqdm import tqdm
        comfy_pbar = ProgressBar(len(timesteps))
        callback = prepare_callback(transformer, steps)

        x_init = latents.clone()
        x_tgt = latents

        # Main sampling loop
        with tqdm(total=len(timesteps)) as progress_bar:
            for idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                if idx < skip_steps:
                    continue
                    
                t_expand = t.repeat(x_init.shape[0])
                
                # Determine current guidance scale based on timestep
                N = len(timesteps)
                if idx < N - drift_steps:
                    current_guidance_scale = target_guidance_scale
                else:
                    current_guidance_scale = drift_guidance_scale

                # Set up source guidance
                source_guidance_expand = (
                    torch.tensor(
                        [source_guidance_scale] * x_init.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(model["dtype"])
                    * 1000.0
                    if source_guidance_scale is not None
                    else None
                )
                
                # Set up target guidance
                target_guidance_expand = (
                    torch.tensor(
                        [current_guidance_scale] * x_init.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(model["dtype"])
                    * 1000.0
                    if current_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type="cuda", dtype=model["dtype"], enabled=True
                ):
                    # Generate noise
                    noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

                    # Sigma values for the current step
                    sigma = t / 1000.0
                    sigma_prev = t_prev / 1000.0

                    # Create source and target noisy latents
                    zt_src = (1-sigma) * x_init + sigma * noise
                    zt_tgt = x_tgt + zt_src - x_init

                    # Forward pass through the model for source condition
                    if idx < N-drift_steps:
                        if riflex_freq_index > 0:
                            # Use freqs directly with RIFLEX
                            vt_src = transformer(
                                zt_src,
                                t_expand,
                                context=source_embeds["prompt_embeds"][0],  
                                context_mask=source_embeds["attention_mask"],
                                clip_context=source_embeds["prompt_embeds_2"],
                                freqs=freqs,
                                guidance=source_guidance_expand,
                                return_dict=True,
                            )["x"]
                        else:
                            # Use freqs_cos and freqs_sin
                            vt_src = transformer(
                                zt_src,
                                t_expand,
                                context=source_embeds["prompt_embeds"][0],
                                context_mask=source_embeds["attention_mask"],
                                clip_context=source_embeds["prompt_embeds_2"],
                                freqs_cos=freqs_cos,
                                freqs_sin=freqs_sin,
                                guidance=source_guidance_expand,
                                return_dict=True,
                            )["x"]
                    else:
                        if idx == N - drift_steps:
                            x_tgt = zt_tgt
                        zt_tgt = x_tgt
                        vt_src = 0

                    # Forward pass through the model for target condition
                    if riflex_freq_index > 0:
                        vt_tgt = transformer(
                            zt_tgt,
                            t_expand,
                            context=target_embeds["prompt_embeds"][0],
                            context_mask=target_embeds["attention_mask"],
                            clip_context=target_embeds["prompt_embeds_2"],
                            freqs=freqs,
                            guidance=target_guidance_expand,
                            return_dict=True,
                        )["x"]
                    else:
                        vt_tgt = transformer(
                            zt_tgt,
                            t_expand,
                            context=target_embeds["prompt_embeds"][0],
                            context_mask=target_embeds["attention_mask"],
                            clip_context=target_embeds["prompt_embeds_2"],
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            guidance=target_guidance_expand,
                            return_dict=True,
                        )["x"]

                    # Compute the velocity delta between target and source
                    v_delta = vt_tgt - vt_src
                
                # Apply the update step
                x_tgt = x_tgt.to(torch.float32)
                v_delta = v_delta.to(torch.float32)
                x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                x_tgt = x_tgt.to(model["dtype"])
                
                # Update progress and handle callbacks
                progress_bar.update()
                if callback is not None:
                    callback_latent = (zt_tgt - vt_tgt * sigma).detach()[-1].permute(1,0,2,3)
                    callback(idx, callback_latent, None, steps)
                else:
                    comfy_pbar.update(1)
                  
        # Reset memory stats and offload if requested
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        return ({
            "samples": x_tgt / VAE_SCALING_FACTOR
        },)
