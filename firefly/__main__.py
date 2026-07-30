"""``python -m firefly`` entry point for installed wheels."""
from firefly._bootstrap import main, prepare_process_streams


prepare_process_streams()

if __name__ == "__main__":
    raise SystemExit(main())
