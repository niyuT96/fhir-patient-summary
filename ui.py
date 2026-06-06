"""
Compatibility entry point for local development.

The main application lives in src.app. Running this file starts the same
Gradio UI so older commands such as `python ui.py` still work.
"""

from src.app import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
