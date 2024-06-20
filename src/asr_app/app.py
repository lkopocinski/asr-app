from pathlib import Path
from tkinter import Tk, filedialog
from typing import Union

import gradio as gr
import whisper

import settings as st
from progress import ProgressListener, create_progress_listener_handle


def on_browse_files():
    root = Tk()
    root.attributes("-topmost", True)
    root.focus_force()
    root.withdraw()

    audio_files_ext = ('*.flac', '*.m4a', '*.mp3', '*.mp4', '*.mpeg',
                       '*.mpga', '*.oga', '*.ogg', '*.wav', '*.webm')
    file_paths = filedialog.askopenfilenames(
        initialdir=st.INITIAL_BROWSE_FILE,
        filetypes=(("Audio files", audio_files_ext),)
    )
    file_paths = "\n".join(file_paths)

    root.destroy()

    return file_paths


def on_transcribe(files_paths: str, model_name: str, progress=gr.Progress()):
    model = whisper.load_model(model_name)

    class FileProgressListener(ProgressListener):
        def __init__(self, file_path: str):
            self.progress_description = f"Trwa przetwarzanie pliku: {file_path}"
            self.finished_message = f"Zakończono przetwarzanie pliku: {file_path}"

        def on_progress(self, current: Union[int, float], total: Union[int, float]):
            progress(progress=(current, total), desc=self.progress_description)

        def on_finished(self):
            gr.Info(message=self.finished_message)

    files_paths = files_paths.split('\n')
    for path in files_paths:
        track_listener = FileProgressListener(file_path=path)
        with create_progress_listener_handle(track_listener):
            result = model.transcribe(str(path), verbose=False)

        save_path = Path(path).with_suffix(".txt")
        save_path.write_text(result["text"])

    return "Zakończono"


with gr.Blocks(title=st.TITLE) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            menu_header = gr.Markdown("""# Menu""")

            with gr.Accordion(label="Pliki do transkrypcji:", open=True):
                input_paths = gr.Markdown()
                browse_button = gr.Button(
                    value="Wybierz pliki",
                    variant="secondary",
                )
                browse_button.click(
                    on_browse_files,
                    outputs=input_paths,
                    show_progress="hidden",
                )

            model_dropdown = gr.Dropdown(
                label="Model",
                choices=['base', 'small', 'medium', 'large'],
                value='small',
            )

            transcribe_button = gr.Button(
                value="Wykonaj transkrypcję",
                variant="primary",
                min_width=1,
            )

        with gr.Column(scale=5):
            header = gr.Markdown("""# Wyniki""")
            output_values = gr.Markdown()

    transcribe_button.click(
        on_transcribe,
        inputs=[input_paths, model_dropdown],
        outputs=[output_values],
    )

if __name__ == '__main__':
    demo.queue().launch()
