from pathlib import Path
from tkinter import Tk, filedialog

import gradio as gr

import whisper


def on_browse():
    root = Tk()
    root.attributes("-topmost", True)
    root.focus_force()
    root.withdraw()

    if file_paths := filedialog.askopenfilenames():
        filenames = [Path(path).stem for path in file_paths]
        filenames = "\n".join(filenames)
    else:
        filenames = ""

    root.destroy()

    return filenames


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                #### Pliki do transkrypcji                
                """
            )
            input_paths = gr.Text(
                show_label=False,
                placeholder="Nie wybrano plików",
                scale=4,
                interactive=False,
            )
            browse_button = gr.Button(
                "Wybierz",
                variant="primary",
                min_width=1
            )
            browse_button.click(
                on_browse,
                outputs=input_paths,
                show_progress="hidden"
            )

        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(choices=['base', 'small', 'medium', 'large'])
            transcribe_button = gr.Button(
                "Wykonaj transkrypcję",
                variant="primary",
                min_width=1
            )
            transcribe_button.click(
                on_transcribe,
                inputs=[input_paths, model_dropdown],
                outputs=[],
                show_progress="hidden"
            )

if __name__ == '__main__':
    demo.launch()
