import argparse


class Opts():
    def __init__(self, args):
        self.n_epoch = args.n_epoch
        self.residual_blocks = args.residual_blocks
        self.lr = args.lr
        self.b1 = args.b1
        self.b2 = args.b2
        self.batch_size = args.batch_size
        self.n_cpu = args.n_cpu
        self.warmup_batches = args.warmup_batches
        self.lambda_adv = args.lambda_adv
        self.lambda_pixel = args.lambda_pixel
        self.pretrained = args.pretrained
        self.dataset_name = args.dataset_name
        self.sample_interval = args.sample_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.hr_height = args.hr_height
        self.hr_width = args.hr_width
        self.channels = args.channels
        self.device = args.device

    def to_dict(self):
        parameters = {
            'n_epoch': self.n_epoch,
            'hr_height': self.hr_height,
            'residual_blocks': self.residual_blocks,
            'lr': self.lr,
            'b1': self.b1,
            'b2': self.b2,
            'batch_size': self.batch_size,
            'n_cpu': self.n_cpu,
            'warmup_batches': self.warmup_batches,
            'lambda_adv': self.lambda_adv,
            'lambda_pixel': self.lambda_pixel,
            'pretrained': self.pretrained,
            'dataset_name': self.dataset_name,
            'sample_interval': self.sample_interval,
            'checkpoint_interval': self.checkpoint_interval,
            'hr_height': self.hr_height,
            'hr_width': self.hr_width,
            'channels': self.channels,
            'device': str(self.device),
        }
        return parameters
