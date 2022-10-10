"""This module implements a simple trainer."""
import sys
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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def train(model, traingenerator, validgenerator, device, output_path, config) :
    """Train the language model and print progression."""
    pad_idx = model.pad_idx
    scaler = GradScaler()
    cross_entropy = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=.1, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=0)
    best_loss = float("Inf")
    nb_batchs = sum(1 for _ in range(0, traingenerator.size, config.batch_size))
    verbose = 0
    with open("training.logs", "a+") as epochs_file:
        try:
            *_, last = enumerate(epochs_file)
            last_epoch, _ = last
            last_epoch = int(last_epoch)
            last_epoch += 1
        except:
            last_epoch = 0
        for epoch in range(last_epoch, last_epoch + config.epochs) :
            loss_sum = 0
            total = 0
            accumulations = 0
            for batch_idx, (X, Y) in tqdm(enumerate(traingenerator(batch_size=config.batch_size)), total=nb_batchs):
                X = torch.tensor(X).to(device)
                Y = torch.tensor(Y).to(device)
                b, s = X.shape
                with autocast(device_type="cuda"):
                    O = model(X) # out.shape = [b, s, vocab_size]
                    loss = cross_entropy(O.view(b * s, -1), Y.view(-1)) # O.shape[0] and Y.shape[0] must be same
                accumulations += b
                verbose += 1
                # accumulate the gradients
                scaler.scale(loss).backward()
                if (accumulations < config.gradients_accumulation) and (batch_idx + config.gradients_accumulation) < nb_batchs:
                    continue
                # if number of gradients accumulations reached then update the parameters
                accumulations = 0
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # if the norm of the gradients vector is superior to 5, then the gradient is so to 5.
                scaler.step(optimizer) # update parameters
                scaler.update()
                # scheduler taking account only the number of time the parameters are update ... 
                # that is the accumulation iterations and not just the number of batchs.
                scheduler.step()
                optimizer.zero_grad()
                loss_sum += loss.item()
                total += 1
                steps_loss = loss_sum / total
                if verbose > config.valid_every_n_batchs:
                    prompted, expected = validgenerator.prompt()
                    print()
                    print(f"epoch={epoch + 1}, train loss={steps_loss}, train ppl={10 ** torch.tensor(steps_loss)} lr={optimizer.param_groups[0]['lr']}")
                    print(f"prompted : {validgenerator.decode(prompted)}")
                    print(f"expected : {validgenerator.decode(expected)}")
                    print(f"generated : {nucleus_sampling(model, validgenerator.tokenizer, prompted, device)}")
                    verbose = 0
            train_loss = loss_sum / total
            epoch_info = f"train loss={train_loss}, train ppl={10 ** torch.tensor(train_loss)}, lr={optimizer.param_groups[0]['lr']}"
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
    LOGGER.info("Loading data generators")
    traingenerator = DataGenerator(config, "train")
    validgenerator = DataGenerator(config, "dev")

    model = TransformerLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == "cuda", """Cannot train on CPU"""
    LOGGER.info(f"Using device {device}")
    model.to(device)
    if config.checkpoint is not None:
        LOGGER.info(f"Loading 'checkpoint {config.checkpoint}'")
        model.load_state_dict(torch.load(config.checkpoint, 
                                         map_location=torch.device(device)))
    LOGGER.info("Training ...")
    train(model=model,
          traingenerator=traingenerator,
          validgenerator=validgenerator,
          device=device,
          output_path=output_path,
          config=config)


if __name__ == "__main__":
    main()
