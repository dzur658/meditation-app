import gradio as gr
import numpy as np
import torch
from kokoro import KModel, KPipeline
from ollama import chat, ChatResponse

# User Defined Variables
OLLAMA_MODEL = "gpt-oss:20b"
TEMPERATURE = 1.0
MAX_TOKENS = 2048

KOKORO_VOICE = "af_nicole"

# run on optimal device depending on gpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

def generate_script(prompt):
    response: ChatResponse = chat(
        model=OLLAMA_MODEL, 
        messages=[
        {
            "role": "system",
            "content": """You are a meditation script generator. \
            Create a calming and soothing meditation script based on the user's prompt. \
            Only output the raw script without any additional commentary. Only output the dialogue of \
            the speaker without anything like pauses or meta actions."""
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    options=[
        "temperature", TEMPERATURE,
        "max_tokens", MAX_TOKENS
    ]
    )
    return response.message.content

def generate_audio(script):
    chunks = [ script[i:i + 500] for i in range(0, len(script), 500) ]

    audio_outputs = []

    for chunk in chunks:
        generator = pipeline(chunk, voice=KOKORO_VOICE)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            audio_outputs.append(audio)
    
    concat_audio = np.concatenate(audio_outputs, axis=0)

    return 26000, concat_audio

# gradio interface
title = "Meditation Session Generator"
description = "Generate a calming meditation session based on your prompt."

demo = gr.Blocks()

user_input = gr.Textbox(label="Enter your meditation prompt here:", lines=4, placeholder="e.g., A guided meditation for stress relief and relaxation.")
generate_button = gr.Button("Generate Meditation Session")
output_audio = gr.Audio(label="Generated Meditation Audio", type="numpy", autoplay=False)

with demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    user_input.render()
    generate_button.render()
    output_audio.render()

    def on_generate_click(prompt):
        script = generate_script(prompt)
        sr, audio = generate_audio(script)
        return (sr, audio)

    generate_button.click(fn=on_generate_click, inputs=user_input, outputs=output_audio)

demo.launch()