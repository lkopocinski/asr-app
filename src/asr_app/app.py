from pathlib import Path
from tkinter import Tk, filedialog
from typing import Union

import gradio as gr
import pandas as pd
import whisper

import settings as st
from progress import ProgressListener, create_progress_listener_handle


def on_browse_dirs():
    root = Tk()
    root.attributes("-topmost", True)
    root.focus_force()
    root.withdraw()

    dir_path = filedialog.askdirectory(
        initialdir=st.INITIAL_BROWSE_DIRS,
    )

    root.destroy()

    return dir_path


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


def on_transcribe(files_paths: str, output_dir: str, model_name: str, progress=gr.Progress()):
    progress(progress=0.33, desc=f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    progress(progress=1, desc=f"Model {model_name} loaded.")

    class FileProgressListener(ProgressListener):
        def __init__(self, file_path: str):
            self.progress_description = f"Trwa przetwarzanie pliku: {file_path}"
            self.finished_message = f"Zakończono przetwarzanie pliku: {file_path}"

        def on_progress(self, current: Union[int, float], total: Union[int, float]):
            progress(progress=(current, total), desc=self.progress_description)

        def on_finished(self):
            gr.Info(message=self.finished_message)

    results = []

    files_paths = files_paths.split('\n')
    for path in files_paths:
        track_listener = FileProgressListener(file_path=path)
        with create_progress_listener_handle(track_listener):
            result = model.transcribe(str(path), verbose=False)

        if Path(output_dir).is_dir():
            save_path = Path(output_dir) / Path(path).name
            save_path = save_path.with_suffix(".txt")
        else:
            save_path = Path(path).with_suffix(".txt")

        save_path.write_text(result["text"])
        results.append((str(path), str(save_path)))

    df = pd.DataFrame(results, columns=['Plik audio', 'Plik tekstowy'])
    return df


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

            with gr.Accordion(label="Folder do zapisu:", open=True):
                output_dir = gr.Markdown()
                browse_button_dir = gr.Button(
                    value="Wybierz folder",
                    variant="secondary",
                )
                browse_button_dir.click(
                    on_browse_dirs,
                    outputs=output_dir,
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
            output_values = gr.DataFrame(
                headers=['Plik audio', 'Plik tekstowy'],
                col_count=(2, "fixed"),
            )

    transcribe_button.click(
        on_transcribe,
        inputs=[input_paths, output_dir, model_dropdown],
        outputs=output_values,
    )

if __name__ == '__main__':
    demo.queue().launch()
