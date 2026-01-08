# meditation-app
A small meditation app made with Gradio that will create a custom meditation for the user.

# Features
- Add your own Ollama Model
- Select any [Kokoro TTS Voice](https://huggingface.co/hexgrad/Kokoro-82M)
- Gradio User Interface

# Get Started
1. Install requirements from requirements.txt
`pip install -r requirements.txt`
2. Ensure [Ollama](https://ollama.com/) is installed on your system, and you have pulled the model you wish to use.
3. Update the user configuration to fit the Ollama model you wish to use, and sampling parameters for your specific model.
3. Run `app.py`
4. Connect to local gradio server (http://127.0.0.1:7860/)!