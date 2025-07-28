import torch
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch
import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import os
from tqdm import tqdm
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import lpips
from skimage.metrics import structural_similarity as ssim

from utils.saicinpainting.training.data.masks import MixedMaskGenerator
from utils.saicinpainting.evaluation.utils import load_yaml
from model import (
    StableDiffusionRevisedInpaintPipeline,
    AsymmetricRevisedAutoencoderKL,
    UNet2DConditionRevisedModel
)

np.set_printoptions(suppress=True)

class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]

class MetricsCalculator:
    def __init__(self, device,ckpt_path="data/ckpt") -> None:
        self.device=device
        # clip
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        # aesthetic model
        self.loss_fn = lpips.LPIPS(net="vgg").eval().to("cuda")
 

    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)
        
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
    def calculate_psnr(self, img_fake, img_real):
        img_fake = np.array(img_fake)
        img_real = np.array(img_real)
        psnr_value = psnr(img_fake, img_real)
        return psnr_value

    
    def calculate_lpips(self, img_pred, img_gt):
        img_fake = lpips.im2tensor(np.array(img_pred)).to("cuda")
        img_real = lpips.im2tensor(np.array(img_gt)).to("cuda")
        lpips_value = self.loss_fn(img_fake, img_real).item()
        return lpips_value

    def calculate_ssim(self, img_fake, img_real):
        img_fake = np.array(img_fake)
        img_real = np.array(img_real)
        ssim_value, _ = ssim(img_fake, img_real, full=True, channel_axis=2)
        return ssim_value
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', 
                        type=str, 
                        default="../paper2/brushbench/mapping_file.json")
    parser.add_argument('--base_dir', 
                        type=str, 
                        default="../paper2/brushbench")
    parser.add_argument('--vae', 
                        type=str, 
                        default="./AsymmetricAutoencoderKL")
    parser.add_argument('--unet', 
                        type=str, 
                        default="./unet")
    parser.add_argument('--saving_root', 
                        type=str, 
                        default="../experiment_data")
    parser.add_argument('--sub_path', 
                        type=str, 
                        default="reproduce2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    config = load_yaml("./utils/saicinpainting/config/random_thick_256.yaml")
    variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
    mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                            variants_n=variants_n)

    with open(args.mapping_file,"r") as f:
        mapping_file=json.load(f)

    masked_image_path = os.path.join(args.saving_root, "masked_images")

    if os.path.exists(os.path.join(args.unet)) and os.path.exists(os.path.join(args.vae)):
        pipe_unet = UNet2DConditionRevisedModel.from_pretrained(args.unet,
                                                        safety_checker=None,
                                                        requires_safety_checker=False).to(device=device, dtype=dtype)

        pipe_vae = AsymmetricRevisedAutoencoderKL.from_pretrained(args.vae).to(device=device, dtype=dtype)
    else:
        print(f"The unet {args.unet} or vae {args.vae} is not found, use the pretrained unet and vae")
        pipe_unet = UNet2DConditionRevisedModel.from_pretrained("SummerRiver5rf/dual_encoder_sd15",
                                                        safety_checker=None,
                                                        requires_safety_checker=False,
                                                        subfolder="unet").to(device=device, dtype=dtype)
        pipe_vae = AsymmetricRevisedAutoencoderKL.from_pretrained("SummerRiver5rf/dual_encoder_sd15",
                                                        subfolder="AsymmetricAutoencoderKL").to(device=device, dtype=dtype)

    pipe = StableDiffusionRevisedInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                            unet = pipe_unet,
                                                            vae = pipe_vae,
                                                            safety_checker=None, 
                                                            requires_safety_checker=False).to(device=device, dtype=dtype)

    
    saving_path = os.path.join(args.saving_root,args.sub_path,"images")
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)    

    saving_path = os.path.join(args.saving_root, args.sub_path)

    for mask_key in ["normal", "pixel"]:   # in pixel and normal
        for key, item in tqdm(mapping_file.items(), desc=args.sub_path+"_"+mask_key):
            image_path=item["image"]
            caption=item["caption"]

            save_path = os.path.join(saving_path,image_path) 
            save_path = save_path.replace(".jpg",f"_{mask_key}.png")
            mask_save_path=os.path.join(args.base_dir,image_path).replace(".jpg",f"_{mask_key}_mask.png")
            
            if os.path.exists(save_path):
                continue

            init_image = np.array(Image.open(os.path.join(args.base_dir,image_path)))

            #Image.open(os.path.join(args.base_dir,image_path)).save("./input.png")
            
            if os.path.exists(mask_save_path):
                mask_image = Image.open(mask_save_path)
                mask = (np.array(mask_image)/255).astype(np.bool_)
            else:
                if mask_key == "pixel":
                    p = random.uniform(0.05,0.95)            
                    mask = np.random.choice([0, 1], size=(512, 512), p=[p, 1-p]).astype(np.bool_)[:,:,np.newaxis]
                else:
                    src_masks = mask_generator.get_masks(init_image)
                    mask_one_indx = np.random.choice(len(src_masks), 1)
                    mask = (src_masks[mask_one_indx[0]]).astype(np.bool_)[:,:,np.newaxis]
                    
                mask_image = Image.fromarray((mask[:,:,0]*255).astype(np.uint8), mode="L").convert("RGB")
                #mask_image.save("./input_mask.png")
                
                if mask_image.height != init_image.shape[0] or mask_image.width != init_image.shape[1]:
                    mask_image = mask_image.resize(init_image.shape[:2], Image.NEAREST)
                mask_image.save(mask_save_path)
        
            init_image = init_image * (~mask)
            init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            #init_image.save("./masked_input.png")
            
            kkk = (image_path.split("/")[-1])[:-4] + f"_{mask_key}.png"
            if not os.path.exists(os.path.join(args.saving_root, masked_image_path, kkk)):
                init_image.save(os.path.join(args.saving_root, masked_image_path, kkk))

            image = pipe(
                prompt=caption, 
                image=init_image, 
                mask_image=mask_image, 
            ).images[0]

            image.save(save_path)
        
        if not os.path.exists(os.path.join(args.saving_root,args.sub_path,f"evaluation_result_sum_{mask_key}.csv")):
            # evaluation
            evaluation_df = pd.DataFrame(columns=['Image ID','PSNR', 'LPIPS', 'SSIM', 'CLIP Similarity']) # 'Image Reward', 'HPS V2.1', 'Aesthetic Score'

            metrics_calculator=MetricsCalculator(device)

            for key, item in tqdm(mapping_file.items()):
                image_path=item["image"]
                prompt=item["caption"]

                save_path = os.path.join(saving_path,image_path) 
                save_path = save_path.replace(".jpg",f"_{mask_key}.png")
                mask_save_path=os.path.join(args.base_dir,image_path).replace(".jpg",f"_{mask_key}_mask.png")

                src_image_path = os.path.join(args.base_dir, image_path)
                src_image = Image.open(src_image_path).resize((512,512))

                tgt_image = Image.open(save_path).resize((512,512))

                evaluation_result=[key]

                #mask = Image.open(mask_save_path).convert("L")
                #mask = (np.array(mask) / 255.).astype(np.bool_)            

                mask = np.ones((512, 512), np.bool_)[:, :, np.newaxis]

                for metric in evaluation_df.columns.values.tolist()[1:]:
                    
                    if metric == 'PSNR':
                        metric_result = metrics_calculator.calculate_psnr(tgt_image, src_image)
                    
                    if metric == 'LPIPS':
                        metric_result = metrics_calculator.calculate_lpips(tgt_image, src_image)
                    
                    if metric == 'SSIM':
                        metric_result = metrics_calculator.calculate_ssim(tgt_image, src_image)
                    
                    if metric == 'CLIP Similarity':
                        metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

                    evaluation_result.append(metric_result)
                
                evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

            print("The averaged evaluation result:")
            averaged_results=evaluation_df.mean(numeric_only=True)
            print(averaged_results)
            
            averaged_results.to_csv(os.path.join(args.saving_root,args.sub_path,f"evaluation_result_sum_{mask_key}.csv"))
            evaluation_df.to_csv(os.path.join(args.saving_root,args.sub_path,f"evaluation_result_{mask_key}.csv"))

            print(f"The generated images and evaluation results is saved in {args.saving_root,args.sub_path}")