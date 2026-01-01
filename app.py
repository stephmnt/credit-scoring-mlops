from fastapi import FastAPI
import gradio as gr

from app.main import app as api_app
from app.main import startup_event
from gradio_app import demo


root_app = FastAPI()
root_app.mount("/api", api_app)
root_app = gr.mount_gradio_app(root_app, demo, path="/")


@root_app.on_event("startup")
def _startup() -> None:
    startup_event()


app = root_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
