import spaces  # must be the first import on ZeroGPU

import tempfile
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

MODELS = {
    "htdemucs": "Best balance. 4 stems: drums, bass, other, vocals.",
    "mdx_extra_q": "Faster and lighter. A bit rougher.",
    "htdemucs_6s": "Adds guitar and piano. Slower. Use short clips.",
}
CACHE = {}


def get_cached(name: str):
    if name not in CACHE:
        model = get_model(name)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        CACHE[name] = model
    return CACHE[name]


@spaces.GPU(duration=180)
def split(audio_path, model_name):
    if not audio_path:
        raise gr.Error("Upload a track first.")
    if model_name not in MODELS:
        model_name = "htdemucs"

    model = get_cached(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None].to(device),
            device=device,
            progress=False,
            split=True,
            overlap=0.25,
        )[0]
    sources = sources * ref.std() + ref.mean()

    out_dir = Path(tempfile.mkdtemp(prefix="stems_"))
    names = list(model.sources)
    files = {}
    for name, stem in zip(names, sources):
        path = out_dir / f"{name}.wav"
        torchaudio.save(str(path), stem.cpu(), model.samplerate)
        files[name] = str(path)

    def pick(*keys):
        for key in keys:
            if key in files:
                return files[key]
        return None

    note = "Done. " + ", ".join(names)
    return (
        pick("vocals"),
        pick("drums"),
        pick("bass"),
        pick("other"),
        pick("guitar"),
        pick("piano"),
        list(files.values()),
        note,
    )


with gr.Blocks(title="Stem Splitter") as demo:
    gr.Markdown(
        "# Stem splitter\n"
        "Demucs on ZeroGPU. Free accounts get about 5 GPU minutes a day, "
        "so keep uploads under about 4 minutes. htdemucs is the default."
    )
    with gr.Row():
        audio = gr.Audio(type="filepath", label="Track")
        model = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="htdemucs",
            label="Model",
            info="htdemucs = quality. mdx_extra_q = faster. htdemucs_6s = guitar/piano.",
        )
    go = gr.Button("Split stems", variant="primary")
    status = gr.Textbox(label="Status")
    with gr.Row():
        vocals = gr.Audio(label="Vocals", type="filepath")
        drums = gr.Audio(label="Drums", type="filepath")
    with gr.Row():
        bass = gr.Audio(label="Bass", type="filepath")
        other = gr.Audio(label="Other", type="filepath")
    with gr.Row():
        guitar = gr.Audio(label="Guitar (6-stem only)", type="filepath")
        piano = gr.Audio(label="Piano (6-stem only)", type="filepath")
    downloads = gr.File(label="Download all stems", file_count="multiple")

    go.click(
        split,
        inputs=[audio, model],
        outputs=[vocals, drums, bass, other, guitar, piano, downloads, status],
    )

if __name__ == "__main__":
    demo.queue().launch()
