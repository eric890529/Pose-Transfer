import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import CLIPVisionModel

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, cond_style = False, **kwargs):
        hs = []
        # with torch.no_grad():
        #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        #     emb = self.time_embed(t_emb)
        #     h = x.type(self.dtype)
        #     for module in self.input_blocks:
        #         h = module(h, emb, context)
        #         hs.append(h)
        #     h = self.middle_block(h, emb, context)

        # if control is not None:
        #     h += control.pop()

        # for i, module in enumerate(self.output_blocks):
        #     if only_mid_control or control is None:
        #         h = torch.cat([h, hs.pop()], dim=1)
        #     else:
        #         h = torch.cat([h, hs.pop() + control.pop()], dim=1)
        #     h = module(h, emb, context)
        enc_cond_emb = cond_style
        mid_cond_emb = cond_style[-1]
        dec_cond_emb = cond_style
        
        # with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, cond = enc_cond_emb[i])
            hs.append(h)
        h = self.middle_block(h, emb, cond = mid_cond_emb)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            if i >= 5:
                h = module(h, emb, cond = dec_cond_emb[-i])
            else:
                h = module(h, emb, cond = dec_cond_emb[-i-1])
            

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            is_contorlNet = True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,is_contorlNet = is_contorlNet
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, is_contorlNet = is_contorlNet
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context=None, cond_style=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        #guided_hint = self.input_hint_block(hint, emb, context)# emb context根本沒吃到??
        guided_hint = self.input_hint_block(hint, emb)# emb context根本沒吃到??

        outs = []

        h = x.type(self.dtype)
        # for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        #     if guided_hint is not None:
        #         h = module(h, emb, context)
        #         h += guided_hint
        #         guided_hint = None
        #     else:
        #         h = module(h, emb, context)
        #     outs.append(zero_conv(h, emb, context))

        # h = self.middle_block(h, emb, context)
        # outs.append(self.middle_block_out(h, emb, context))
        enc_cond_emb = cond_style
        mid_cond_emb = cond_style[-1]
        #dec_cond_emb = cond_style
        
        for i, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if guided_hint is not None:
                h = module(h, emb, cond = enc_cond_emb[i])
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, cond = enc_cond_emb[i])
            outs.append(zero_conv(h, emb, cond = enc_cond_emb[i]))

        h = self.middle_block(h, emb, cond = mid_cond_emb)
        outs.append(self.middle_block_out(h, emb, cond = mid_cond_emb))
        
        # for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        #     if guided_hint is not None:
        #         h = module(h, emb)
        #         h += guided_hint
        #         guided_hint = None
        #     else:
        #         h = module(h, emb)
        #     outs.append(zero_conv(h, emb))

        # h = self.middle_block(h, emb)
        # outs.append(self.middle_block_out(h, emb))

        return outs

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, style_encoder_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        self.style_encoder = instantiate_from_config(style_encoder_config) # style_encoder
        
        # self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        # self.clip_encoder.requires_grad_(False)
        # self.adapter = Embedding_Adapter()

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, is_inference=False, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, is_inference=is_inference, *args, **kwargs)#抓到 latent code of x
        #control = batch[self.control_key] #抓到condition的圖片
        if not is_inference:
            control = torch.cat([batch['source_image'], batch['target_image']], 0)
            style_img = control
            ## 這裡做pose跟圖片concat?? pose也要經過encoder? 還是concat完再進入encoder?
            pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)
            #control = torch.cat([control,pose] , dim=1) # 再看看要pose就好 還是全部都要一起餵入
        else:
            control = batch['source_image']
            style_img = control
            ## 這裡做pose跟圖片concat?? pose也要經過encoder? 還是concat完再進入encoder?
            pose = batch['target_skeleton']
        control = torch.cat([control,pose] , dim=1)
        
        
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        #control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control], c_style = [style_img])
        #return x, dict(c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # if cond['c_crossattn'][0] is not None:
        #     cond_txt = torch.cat(cond['c_crossattn'], 1)
        # else:
        #     cond_txt = None
        
        # cond_style = self.style_encoder(torch.cat(cond['c_style'], 1)) #later
        # encoder_posterior_list = []
        # for c in cond_style:
        #     encoder_posterior_list.append(c)
        # z_style_list = []
        # for encoder_posterior in encoder_posterior_list:
        #     z_style_list.append(self.get_first_stage_encoding(encoder_posterior).detach())
        # cond_style = z_style_list
        
        from torchvision.transforms import Resize 
        torch_resize = Resize([224,224])
        ### GET Style encoder features ###
        # from torchvision.transforms import Resize 
        # torch_resize = Resize([32,32])
        # cond['c_style'][0] = torch_resize(cond['c_style'][0])
        # cond_style = self.style_encoder(cond['c_style'][0])
        
        encoder_posterior = self.encode_first_stage(cond['c_style'][0].to("cuda"))
        z_style = self.get_first_stage_encoding(encoder_posterior).detach() # latent code
        cond_style = self.style_encoder(z_style)
        
        # Get CLIP embeddings
        controlInput = torch.cat(cond['c_concat'], 1)
        controlStyle, controlPose = torch.split(controlInput, [3,20], dim = 1) #可以check是否跟c_style一樣
        # inputs = {"pixel_values": torch_resize(controlStyle).to(self.device)}
        # clip_hidden_states =  self.clip_encoder(**inputs).last_hidden_state.to(self.device)
        
        # # Get VAE embeddings
        # image = controlStyle.to(device=self.device, dtype=self.dtype)
        # encoder_posterior = self.encode_first_stage(image)
        
        ############################## vae_hidden_states = self.get_first_stage_encoding(encoder_posterior).detach() # latent code #可以check是否跟cond_style一樣
        
        ## Get Adapter to controlNet
        # encoder_hidden_states = self.adapter(clip_hidden_states, z_style)
        
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            #control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            
            # for test vram佔滿的問題
            # controlInput = torch.cat(cond['c_concat'], 1)
            # _, tmp = torch.split(controlInput, [3,20], dim = 1)
            # if controlInput.shape[0] == 4:
            #     cond_txt = torch.rand(4,50,768).to(device=self.device)
            # else:
            #     cond_txt = torch.rand(controlInput.shape[0],50,768).to(device=self.device)
                
            # control = self.control_model(x=x_noisy, hint=tmp, timesteps=t, context=cond_txt)
            
            control = self.control_model(x=x_noisy, hint = controlPose, timesteps=t, cond_style = cond_style)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            
            # control = None
            #eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            eps = diffusion_model(x=x_noisy, timesteps=t, control=control, only_mid_control=self.only_mid_control, cond_style = cond_style) # , style_encoder = style_encoder

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=6, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        #c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        c_cat = c["c_concat"][0][:N]
        c_style = c["c_style"][0][:N] #
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat #* 2.0 - 1.0
        #log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            # samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
            #                                          batch_size=N, ddim=use_ddim,
            #                                          ddim_steps=ddim_steps, eta=ddim_eta)
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_style =  c_style
            uc_full = {"c_concat": [uc_cat],  "c_style" : [uc_style]}# "c_crossattn": [uc_cross],
            # samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
            #                                  batch_size=N, ddim=use_ddim,
            #                                  ddim_steps=ddim_steps, eta=ddim_eta,
            #                                  unconditional_guidance_scale=unconditional_guidance_scale,
            #                                  unconditional_conditioning=uc_full,
            #                                  )
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_style" : [c_style]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        transform_param = self.extractAttnParam(self.model.diffusion_model.input_blocks)
        params =  list(self.control_model.parameters() )+ list(self.style_encoder.parameters())  
              # #list(self.first_stage_model.parameters()) +  list(self.adapter.parameters())
        for p in transform_param:
            params += list(p)

        if not self.sd_locked:
            # params += list(self.model.parameters())
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def extractAttnParam(self, model):
        spatial_transformer_params = []
        # 遍历模型的所有子模块，并检查它们是否为 SpatialTransformer 的实例
        for name, module in model.named_modules():
            if isinstance(module, SpatialTransformer):
                # 如果是 SpatialTransformer 实例，则将其参数添加到列表中
                spatial_transformer_params.extend([module.parameters()])
        return spatial_transformer_params
    
    def on_train_epoch_start(self):
        if self.current_epoch > 40:
            self.set_optim_lr(self.optimizers(), 1e-6)
        if self.current_epoch > 60:
            self.set_optim_lr(self.optimizers(), 5e-7)
        if self.current_epoch > 90:    
            self.set_optim_lr(self.optimizers(), 2.5e-7)

    def on_train_epoch_end(self):
        if self.current_epoch > 40:
            self.set_optim_lr(self.optimizers(), 1e-6)
        if self.current_epoch > 60:
            self.set_optim_lr(self.optimizers(), 5e-7)
        if self.current_epoch > 90:    
            self.set_optim_lr(self.optimizers(), 2.5e-7)
            
    def set_optim_lr( self, optim, lr ) :
        for param in optim.param_groups :
            param['lr'] = lr
            
    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

class BeatGANsEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        use_style_condition=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ##
        self.input_num_blocks = [0 for _ in range(len(self.channel_mult))]
        self.input_num_blocks[0] = 1
        ##
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_style_condition=use_style_condition
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    # if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                    #     layers.append(
                    #         AttentionBlock(
                    #             ch,
                    #             use_checkpoint=use_checkpoint,
                    #             num_heads=num_heads,
                    #             num_head_channels=dim_head,
                    #             use_new_attention_order=use_new_attention_order,
                    #         ) if not use_spatial_transformer else SpatialTransformer(
                    #             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                    #             disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                    #             use_checkpoint=use_checkpoint
                    #         )
                    #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                ##
                self.input_num_blocks[level] += 1
                ##
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_style_condition=use_style_condition
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ##
                self.input_num_blocks[level + 1] += 1
                ##
                ds *= 2
                self._feature_size += ch

        # self._to_vector_layers = [nn.Sequential(
        #         normalization(ch),
        #         nn.SiLU(),
        #         nn.AdaptiveAvgPool2d((1, 1)),
        #         conv_nd(conf.dims, ch, ch, 1),
        #         nn.Flatten(),
        #         ).cuda() for ch in self._feature_size]

    def forward(self, x, t=None, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # hs = []
        hs = [[] for _ in range(len(self.channel_mult))]
        #emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        results = []
        
        # for module in self.input_blocks:
        #     h = module(h, emb, cond = cond_style)
        #     hs.append(h)
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=None)
                # print(i, j, h.shape)
                hs[i].append(h)
                results.append(h)
                #print (h.shape)
                k += 1
        assert k == len(self.input_blocks)

        # vectors = []

        # for i, feat in enumerate(results):
        #     vectors.append(self._to_vector_layers[i](feat))

        return results #把每一層的特徵抓出來?
    
class Embedding_Adapter(nn.Module):
    def __init__(self, input_nc=38, output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter, self).__init__()

        self.save_method_name = "adapter"

        #self.pool =  nn.MaxPool2d(2)
        self.vae2clip = nn.Linear(1024, 768)

        self.linear1 = nn.Linear(54, 50) # 50 x 54 shape

        # initialize weights
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(torch.eye(50, 54))

        if chkpt is not None:
            pass

    def forward(self, clip, vae):
        
        #vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 40 32 --> 1 4 1280

        vae = self.vae2clip(vae) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        # Encode

        concat = rearrange(concat, 'b c d -> b d c')
        concat = self.linear1(concat)
        concat = rearrange(concat, 'b d c -> b c d')

        return concat
