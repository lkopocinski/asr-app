from pathlib import Path

import click
import whisper


@click.command()
@click.option("--file_path", type=click.Path(path_type=Path))
@click.option("--model_name")
def main(file_path: Path, model_name: str):
    model = whisper.load_model(model_name)
    result = model.transcribe(str(file_path))

    save_path = file_path.with_suffix(".txt")
    save_path.write_text(result["text"])


if __name__ == '__main__':
    main()
