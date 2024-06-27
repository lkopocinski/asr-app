from tkinter import Tk, filedialog

from asr_app.settings import settings


def browse_dir() -> str:
    root = Tk()
    root.attributes("-topmost", True)
    root.focus_force()
    root.withdraw()

    dir_path = filedialog.askdirectory(
        initialdir=settings.browse_dir_initial_dir,
    )

    root.destroy()

    return dir_path


def browse_audio_files() -> tuple[str]:
    root = Tk()
    root.attributes("-topmost", True)
    root.focus_force()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        initialdir=settings.browse_files_initial_dir,
        filetypes=(("Audio files", settings.audio_files_ext),)
    )

    root.destroy()

    return file_paths


def browse_audio_files_str() -> str:
    file_paths = browse_audio_files()
    return "\n".join(file_paths)
