# ckpt2sample_gen


1. ckpt 2 diffusers

``` sh
convert_original_stable_diffusion_to_diffusers.py --checkpoint_path <.ckpt path> --original_config_file v1-inference.yaml --dump_path <diffusers output dir path>
```

2. prompt2wav

gradio interface

``` sh
python gen.py <prompt string> <wav out filepath> <diffuser model path>
```

