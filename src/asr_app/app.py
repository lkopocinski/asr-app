from pathlib import Path

from typing import Union

import gradio as gr
import pandas as pd
import whisper

from settings import settings
from asr_app.io import browse_audio_files_str, browse_dir
from asr_app.progress import ProgressListener, create_progress_listener_handle
import texts as t


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


with gr.Blocks(title=t.title) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            menu_header = gr.Markdown(t.menu_header)

            with gr.Accordion(label=t.files_label, open=True):
                input_paths = gr.Markdown()
                browse_button = gr.Button(
                    value=t.browse_files_btn,
                    variant="secondary",
                )
                browse_button.click(
                    browse_audio_files_str,
                    outputs=input_paths,
                    show_progress="hidden",
                )

            with gr.Accordion(label=t.dir_label, open=True):
                output_dir = gr.Markdown()
                browse_button_dir = gr.Button(
                    value=t.browse_dir_btn,
                    variant="secondary",
                )
                browse_button_dir.click(
                    browse_dir,
                    outputs=output_dir,
                    show_progress="hidden",
                )

            model_dropdown = gr.Dropdown(
                label=t.model_dropdown_label,
                choices=settings.whisper_models_names,
                value=settings.whisper_default_model,
            )

            transcribe_button = gr.Button(
                value=t.transcribe_btn,
                variant="primary",
                min_width=1,
            )

        with gr.Column(scale=5):
            header = gr.Markdown(t.results_header)
            output_values = gr.DataFrame(
                headers=t.results_table_header,
                col_count=(2, "fixed"),
            )

    transcribe_button.click(
        on_transcribe,
        inputs=[input_paths, output_dir, model_dropdown],
        outputs=output_values,
    )


def main():
    demo.queue().launch()
