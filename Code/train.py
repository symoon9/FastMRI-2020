import argparse
import shutil
from pathlib import Path
import os
# from utils.learning.train_part import Trainer
from utils.common.arguments import parse
from utils.common.utils import make_dir, yaml_to_dict


if __name__ == '__main__':
    # get arguments
    args = parse()
    # make dir for checkpoints and results
    args = make_dir(args)
    # save configs to result dir
    script = "run_script"
    if args.use_kspace:
        script = script+'_kspace'

    os.system(f'cp {args.code_dir}/{script}.sh {args.result_path}/{script}.sh')
    # start training
    lib = 'utils.learning.train_part'
    if args.use_kspace:
        lib = lib+'_kspace'
    trainer = getattr(__import__(lib, fromlist=[""]), "Trainer")(args)
    model = trainer.train()
    # make test recon files
    trainer.test(model)