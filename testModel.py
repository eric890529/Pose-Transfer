from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from cldm.model import create_model, load_state_dict

x = create_model("./models/test.yaml")
resume_path = "./testmodel/diffusion_pytorch_model.bin"
x.load_state_dict(load_state_dict(resume_path, location='cpu'))#可以在load dict+一個參數 硬load ## attention這樣好像不會更新到? , strict=False