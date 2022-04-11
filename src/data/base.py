"""Base module of custom datamodules."""


import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):
    """Abstract base class for custom datamodules."""

    def __init__(self):
        super().__init__()


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()