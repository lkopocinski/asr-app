from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    browse_files_initial_dir: str
    browse_dir_initial_dir: str

    audio_files_ext: tuple[str] = ('*.flac', '*.m4a', '*.mp3', '*.mp4', '*.mpeg',
                                   '*.mpga', '*.oga', '*.ogg', '*.wav', '*.webm')

    whisper_models_names = ['base', 'small', 'medium', 'large']
    whisper_default_model = 'small'


settings = Settings()


