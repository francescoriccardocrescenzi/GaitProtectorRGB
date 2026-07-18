import os
import cv2
import numpy as np
import torch
from diffusers import VideoToVideoSDPipeline
from diffusers.pipelines.deprecated.text_to_video_synthesis.pipeline_text_to_video_synth_img2img import retrieve_latents
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor

VIDEO_PATH = "data/sample.mp4"
OUTPUT_DIR = "data"
NUM_FRAMES = 16
RESOLUTION = 256
MODEL_FPS = 8
NOISE_STRENGTH = 0.5
NUM_INFERENCE_STEPS = 20

# Utils

def load_video_frames(path, num_frames, size, target_fps):
    cap = cv2.VideoCapture(path)
    orig_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    stride = max(round((cap.get(cv2.CAP_PROP_FPS) or target_fps) / target_fps), 1)
    wanted = set(range(0, stride * num_frames, stride))

    frames, idx = [], 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, orig_size

def save(frames, name, orig_size):
    frames = [cv2.resize(f, orig_size) for f in frames]
    path = os.path.join(OUTPUT_DIR, name)
    export_to_video(frames, path, fps=MODEL_FPS, macro_block_size=1)

def as_frames(video):
    """(b, c, f, h, w) -> (b*f, c, h, w), the shape pipe.vae/pipe.unet operate on."""
    b, c, f, h, w = video.shape
    return video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w), (b, f)

def as_video(frames, bf):
    b, f = bf
    c, h, w = frames.shape[1:]
    return frames.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)

# Diffusion pipeline

def load_pipeline():
    pipe = VideoToVideoSDPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()
    return pipe

def encode(pipe, video, generator):
    frames, bf = as_frames(video)
    # The scaling factor brings vae features to unit variance
    latents = retrieve_latents(pipe.vae.encode(frames), generator=generator) * pipe.vae.config.scaling_factor
    return as_video(latents, bf)

def inject_noise(pipe, latents, generator, device):
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps, _ = pipe.get_timesteps(NUM_INFERENCE_STEPS, NOISE_STRENGTH, device)
    noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])
    return latents, timesteps

def denoise(pipe, latents, timesteps, generator, device):
    prompt_embeds, _ = pipe.encode_prompt("", device, 1, do_classifier_free_guidance=False)
    for t in timesteps:
        latent_input = pipe.scheduler.scale_model_input(latents, t)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

        lat_frames, bf = as_frames(latents)
        pred_frames, _ = as_frames(noise_pred)
        lat_frames = pipe.scheduler.step(pred_frames, t, lat_frames, generator=generator).prev_sample
        latents = as_video(lat_frames, bf)
    return latents

def decode(pipe, latents):
    frames, bf = as_frames(latents / pipe.vae.config.scaling_factor)
    frames = pipe.vae.decode(frames).sample.float()
    video = as_video(frames, bf)
    return pipe.video_processor.postprocess_video(video=video, output_type="np")[0]

# Main

@torch.no_grad()
def main():
    pipe = load_pipeline()
    device = pipe._execution_device
    generator = None

    raw_frames, orig_size = load_video_frames(VIDEO_PATH, NUM_FRAMES, RESOLUTION, MODEL_FPS)
    video = pipe.video_processor.preprocess_video(raw_frames).to(device=device, dtype=torch.float16)
    save(pipe.video_processor.postprocess_video(video=video.float(), output_type="np")[0], "input.mp4", orig_size)

    latents = encode(pipe, video, generator)

    latents, timesteps = inject_noise(pipe, latents, generator, device)
    save(decode(pipe, latents), "noised.mp4", orig_size)

    latents = denoise(pipe, latents, timesteps, generator, device)
    save(decode(pipe, latents), "reconstructed.mp4", orig_size)


if __name__ == "__main__":
    main()
