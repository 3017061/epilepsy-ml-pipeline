import typer
app = typer.Typer(help="Epilepsy ML Pipeline")

from . import pipeline
app.add_typer(pipeline.app, name="pipeline")

def main():
    app()

if __name__ == "__main__":
    main()
