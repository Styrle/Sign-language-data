"""BSL Data Extraction CLI."""

import click


@click.group()
def main():
    """BSL Data Extraction Tool - Process and normalize BSL landmark data."""
    pass


@click.command()
def process_all():
    """Run the full processing pipeline."""
    click.echo("Processing pipeline not yet implemented.")
    click.echo("This will run: import -> normalize -> angles -> validate -> generate")


main.add_command(process_all, name="process-all")


if __name__ == "__main__":
    main()
