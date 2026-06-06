"""Entry point for running `python -m src`."""

from src.start import maybe_load_sample_bundle


if __name__ == "__main__":
    maybe_load_sample_bundle()

    from src.app import demo

    demo.launch(server_name="0.0.0.0", server_port=7860)
