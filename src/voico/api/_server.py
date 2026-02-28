import argparse
import sys


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install voico[server]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Voico API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    from .app import create_app
    app = create_app(db_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
