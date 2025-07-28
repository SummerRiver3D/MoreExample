# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from diffusers.models.autoencoders.autoencoder_asym_kl import *
from diffusers.models.autoencoders.vae import *

class MaskConditionRevisedDecoder(nn.Module):
    r"""The `MaskConditionRevisedDecoder` should be used in combination with [`AsymmetricAutoencoderKL`] to enhance the model's
    decoder with a conditioner on the mask and masked image.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # condition encoder
        self.condition_encoder = MaskConditionEncoder(
            in_ch=out_channels,
            out_ch=block_out_channels[0],
            res_ch=block_out_channels[-1],
        )

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        z: torch.FloatTensor,
        image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        latent_embeds: Optional[torch.FloatTensor] = None,
        return_latent_state: bool = False,
    ) -> torch.FloatTensor:
        r"""The forward method of the `MaskConditionRevisedDecoder` class."""
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # condition encoder
                if image is not None and mask is not None:
                    masked_image = (1 - mask) * image
                    im_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.condition_encoder),
                        masked_image,
                        mask,
                        use_reentrant=False,
                    )
                    
                if return_latent_state:
                    return im_x[str(tuple(sample.shape))].clone()
                
                # up
                for up_block in self.up_blocks:
                    if image is not None and mask is not None:
                        sample_ = im_x[str(tuple(sample.shape))]
                        mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode='nearest')
                        sample = sample * mask_ + sample_ * (1 - mask_)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
                if image is not None and mask is not None:
                    sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # condition encoder
                if image is not None and mask is not None:
                    masked_image = (1 - mask) * image
                    im_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.condition_encoder),
                        masked_image,
                        mask,
                    )
                    
                if return_latent_state:
                    return im_x[str(tuple(sample.shape))].clone()
                # up
                
                for up_block in self.up_blocks:
                    if image is not None and mask is not None:
                        sample_ = im_x[str(tuple(sample.shape))]
                        mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                        sample = sample * mask_ + sample_ * (1 - mask_)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
                if image is not None and mask is not None:
                    sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # condition encoder
            if image is not None and mask is not None:
                masked_image = (1 - mask) * image
                im_x = self.condition_encoder(masked_image, mask)

            if return_latent_state:
                return im_x[str(tuple(sample.shape))].clone()
            
            # up
            for up_block in self.up_blocks:
                if image is not None and mask is not None:
                    sample_ = im_x[str(tuple(sample.shape))]
                    mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                    sample = sample * mask_ + sample_ * (1 - mask_)
                sample = up_block(sample, latent_embeds)
            if image is not None and mask is not None:
                sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AsymmetricRevisedAutoencoderKL(ModelMixin, ConfigMixin):
    r"""
    Designing a Better Asymmetric VQGAN for StableDiffusion https://arxiv.org/abs/2306.04632 . A VAE model with KL loss
    for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        down_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of down block output channels.
        layers_per_down_block (`int`, *optional*, defaults to `1`):
            Number layers for down block.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        up_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of up block output channels.
        layers_per_up_block (`int`, *optional*, defaults to `1`):
            Number layers for up block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        norm_num_groups (`int`, *optional*, defaults to `32`):
            Number of groups to use for the first normalization layer in ResNet blocks.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        down_block_out_channels: Tuple[int, ...] = (64,),
        layers_per_down_block: int = 1,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        up_block_out_channels: Tuple[int, ...] = (64,),
        layers_per_up_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=down_block_out_channels,
            layers_per_block=layers_per_down_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = MaskConditionRevisedDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=up_block_out_channels,
            layers_per_block=layers_per_up_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        self.register_to_config(block_out_channels=up_block_out_channels)
        self.register_to_config(force_upcast=False)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[torch.FloatTensor]]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.FloatTensor,
        image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        return_latent_state: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, image, mask, return_latent_state=return_latent_state)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        return_latent_state: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        decoded = self._decode(z, image, mask, return_latent_state=return_latent_state).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            mask (`torch.FloatTensor`, *optional*, defaults to `None`): Optional inpainting mask.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, sample, mask).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
