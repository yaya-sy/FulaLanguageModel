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

import torch
from torch import nn
from tqdm import tqdm
import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def train(model, traingenerator, validgenerator, device, output_path, config) :
    """Train the language model and print progression."""
    pad_idx = model.pad_idx
    cross_entropy = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_loss = float("Inf")
    nb_batchs = sum(1 for _ in range(0, traingenerator.size, config.batch_size))
    verbose = 0
    for epoch in range(config.epochs) :
        loss_sum = 0
        total = 0
        for X, Y in tqdm(traingenerator(batch_size=config.batch_size), total=nb_batchs) :
            model.zero_grad()
            X = torch.tensor(X).to(device)
            Y = torch.tensor(Y).to(device)
            b, s = X.shape
            O = model(X) # out.shape = [b, s, vocab_size]
            loss = cross_entropy(O.view(b * s, -1), Y.view(-1)) # O.shape[0] and Y.shape[0] must be same
            loss.backward() # backprobagation in order to compute the gradients of the loss function wrt parameters
            optimizer.step() # update parameters
            loss_sum += loss.item()
            total += 1
            if verbose > config.print_generation_steps:
                prompted, expected = validgenerator.prompt()
                print(f"prompted : {validgenerator.decode(prompted)}")
                print(f"expected : {validgenerator.decode(expected)}")
                print(f"generated : {nucleus_sampling(model, validgenerator.tokenizer, prompted, device)}")
                print()
                verbose = 0
            verbose += 1
        train_loss = loss_sum / total
        print(f"epoch={epoch + 1}, train loss={train_loss}, train ppl={10 ** torch.tensor(train_loss)} lr={optimizer.param_groups[0]['lr']}")
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
    print(f"Parameters:{config}")
    LOGGER.info("Loading data generators")
    traingenerator = DataGenerator(config, "train")
    validgenerator = DataGenerator(config, "dev")

    model = TransformerLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device {device}")
    model.to(device)
    LOGGER.info("Training ...")
    train(model=model,
          traingenerator=traingenerator,
          validgenerator=validgenerator,
          device=device,
          output_path=output_path,
          config=config)


if __name__ == "__main__":
    main()


