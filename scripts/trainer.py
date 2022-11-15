"""This module implements a simple trainer."""
import sys
from turtle import clear

from numpy import gradient, isin
sys.path.append('.')
from data_generator import DataGenerator
from generator import nucleus_sampling
from scripts.config.config import Config
from scripts.model.language_model import TransformerLM
from argparse import ArgumentParser
from pathlib import Path
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau # CosineAnnealingLR
from tqdm import tqdm
import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_optimizer(model):
    """"""
    no_decay = set()
    decay = set()
    no_weight_decay = nn.LayerNorm
    weight_decay = nn.Linear
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            # custom nn.Parameters named embeddings
            if not mn and ("embeddings" in fpn or "radius" in fpn):
                no_decay.add(fpn)
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, no_weight_decay):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, weight_decay):
                decay.add(fpn)
    param_dict = dict(model.named_parameters())
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups
    
def train(model, traingenerator, validgenerator, device, output_path, config) :
    """Train the language model and print progression."""
    pad_idx = model.word_padding_idx
    optim_groups = get_optimizer(model)
    cross_entropy = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(optim_groups, lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,mode="min", factor=0.8, patience=2, threshold=0.001)
    # scheduler = CosineAnnealingLR(optimizer, T_max=config.step_size_up, eta_min=0)
    best_loss = float("Inf")
    nb_batchs = sum(1 for _ in range(0, traingenerator.size, config.batch_size))
    verbose = 0
    with open("training.logs", "a+") as epochs_file:
        try:
            *_, last, _ = enumerate(epochs_file)
            last_epoch, _ = last
            last_epoch = int(last_epoch)
            last_epoch += 1
        except:
            last_epoch = 0
        for epoch in range(last_epoch, last_epoch + config.epochs) :
            loss_sum = 0
            total = 0
            for batch_idx, (X, Y) in tqdm(enumerate(traingenerator(batch_size=config.batch_size), 1), total=nb_batchs):
                X = torch.tensor(X).to(device)
                Y = torch.tensor(Y).to(device)
                b, s = X.shape
                O = model(X) # out.shape = [b, s, vocab_size]
                loss = cross_entropy(O.view(b * s, -1), Y.view(-1)) # O.shape[0] and Y.shape[0] must be same
                verbose += 1
                # accumulate the gradients
                loss.backward()
                # if number of gradients accumulations reached then update the parameters
                if config.gradients_accumulation is not None:
                    if batch_idx % config.gradients_accumulation != 0 and (nb_batchs - batch_idx) > config.gradients_accumulation:
                        continue
                if config.norm_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.norm_clip) # clip gradient vectors by norm
                optimizer.step() # update parameters
                # scheduler.step()
                loss_sum += loss.item()
                total += 1
                steps_loss = loss_sum / total
                if verbose > config.valid_every_n_batchs:
                    prompted, expected = validgenerator.prompt()
                    print()
                    print(f"epoch={epoch + 1}, train loss={steps_loss}, train ppl={10 ** torch.tensor(steps_loss)} lr={optimizer.param_groups[0]['lr']}, radius={model.radius.item()}")
                    print(f"prompted : {validgenerator.decode(prompted)}")
                    print(f"expected : {validgenerator.decode(expected)}")
                    print(f"generated : {nucleus_sampling(model, validgenerator.tokenizer, prompted, device)}")
                    verbose = 0
                model.zero_grad()
            train_loss = loss_sum / total
            # scheduler taking account only the number of time the parameters are update ... 
            # that is the accumulation iterations and not just the number of batchs.
            scheduler.step(train_loss)
            epoch_info = f"train loss={train_loss}, train ppl={10 ** torch.tensor(train_loss)}, lr={optimizer.param_groups[0]['lr']}, radius={model.radius.item()}"
            epochs_file.write(epoch_info + "\n")
            print()
            print(f"epoch={epoch + 1}, {epoch_info}")
            if train_loss < best_loss :
                best_loss = train_loss
                torch.save(model.state_dict(), output_path / "fula.pt")

def main():
    """Parse arguments and run training."""
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The YAML config file.")
    parser.add_argument("-o", "--output_directory", help="Where the checkpoints will be saved.")

    args = parser.parse_args()
    output_path = Path(args.output_directory)
    output_path.mkdir(exist_ok=True, parents=True)
    with open(args.config, "r") as config_file:
        yaml_config = yaml.safe_load(config_file)
    config = Config(**yaml_config)
    print(f"Parameters={config}")

    model = TransformerLM(config)
    get_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device {device}")
    model.to(device)
    if config.checkpoint is not None:
        LOGGER.info(f"Loading 'checkpoint {config.checkpoint}'")
        model.load_state_dict(torch.load(config.checkpoint, 
                                         map_location=torch.device(device)))
    LOGGER.info("Loading data generators")
    traingenerator = DataGenerator(config, "train")
    validgenerator = DataGenerator(config, "dev")
    LOGGER.info("Training ...")
    train(model=model,
          traingenerator=traingenerator,
          validgenerator=validgenerator,
          device=device,
          output_path=output_path,
          config=config)


if __name__ == "__main__":
    main()