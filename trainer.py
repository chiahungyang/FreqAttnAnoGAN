"""Train model using the framework of Pytorch Lightning CLI."""


import os
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY
from src.models import BaseAnoGenAdvNet
from src.data import BaseDataModule


PRETRAIN_DIR = './models'


class CustomCLI(LightningCLI):
    """Custom CLI extended from Pytorch Lightning."""

    def add_arguments_to_parser(self, parser):
        # Add additional arguments
        parser.add_argument(
            '--name',
            type=str,
            default='default',
            help='Experiment name used for the saved model.'
        )
        # Link arguments
        parser.link_arguments('name', 'trainer.logger.init_args.name')
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key='generator',
            link_to='model.gen_opt_init'
        )
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key='discriminator',
            link_to='model.disc_opt_init'
        )
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key='encoder',
            link_to='model.enc_opt_init'
        )
    
    def before_fit(self):
        if self.model.training_phase == 'encoder':  # when training encoder
            self._load_pretrained_model()

    def after_fit(self):
        # Output best checkpoint as the pretrained model
        torch.save(
            type(self.model).load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path
            ),
            os.path.join(PRETRAIN_DIR, self.config['name'] + '.pth')
        )
    
    def before_test(self):
        self._load_pretrained_model()
    
    def before_predict(self):
        self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        self.model.load_state_dict(torch.load(
            os.path.join(PRETRAIN_DIR, self.config['name'] + '.pt')
        ).state_dict())



if __name__ == '__main__':
    cli = CustomCLI(
        BaseAnoGenAdvNet,
        BaseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
