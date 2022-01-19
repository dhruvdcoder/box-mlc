import logging
import os
import sys
import click

if os.environ.get("BOX_MLC_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("BOX_MLC_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


@click.command()
def main(args=None):
    """Console script for box_mlc."""
    click.echo("Replace this message by putting your code into "
               "box_mlc.__main__.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")

    return 0

if __name__ == "__main__":
    main(prog_name="Box MLC")
