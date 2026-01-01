from app_entry import app, demo  # re-export for HF Spaces


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
