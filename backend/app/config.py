from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://geodetect:geodetect@localhost:5432/geodetect"
    SECRET_KEY: str = "change-me-in-production"
    DEBUG: bool = False
    UPLOAD_DIR: str = "uploads"
    MODEL_CHECKPOINT: str = "ml/checkpoints/best_model.pt"
    RF_PATH: str = "ml/checkpoints/rf_model.joblib"

    class Config:
        env_file = ".env"


settings = Settings()
