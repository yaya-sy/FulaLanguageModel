"""This module implements a simple trainer."""
import sys
sys.path.append('.')
from batch_generator import BatchGenerator
from generator import nucleus_sampling
from scripts.config.config import Config
from scripts.model.language_model import TransformerLM
from argparse import ArgumentParser

import torch
from torch import nn
from tqdm import tqdm
import yaml

def train(model, traingenerator, validgenerator, device, batch_size=6, epochs=3, lr=0.00036) :
    """Train the language model and print progression."""
    pad_idx = model.pad_idx
    cross_entropy = nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("Inf")
    nb_batchs = sum(1 for _ in range(0, traingenerator.size, batch_size))
    verbose = 5_000
    for epoch in range(epochs) :
        loss_sum = 0
        total = 0
        for X, Y in tqdm(traingenerator(batch_size=batch_size), total=nb_batchs) :
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
            verbose += 1
            if verbose > 5_000:
                prompted, expected = validgenerator.prompt()
                print(f"prompted : {validgenerator.decode(prompted)}")
                print(f"expected : {validgenerator.decode(expected)}")
                print(f"generated : {nucleus_sampling(model, validgenerator.tokenizer, prompted, device)}")
                print("-----" * 20)
                verbose = 0
        train_loss = loss_sum / total
        print(f"epoch={epoch + 1}, train loss={train_loss}, train ppl={10 ** torch.tensor(train_loss)} lr={optimizer.param_groups[0]['lr']}")
        if train_loss < best_loss :
            best_loss = train_loss
            # torch.save(model.state_dict(), "fula_fr.pt")

def main():
    """Parse arguments and run training."""
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The YAML config file.")

    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        yaml_config = yaml.safe_load(config_file)
    config = Config(vocab_size=yaml_config["vocab_size"],
                    train=yaml_config["train"],
                    dev=yaml_config["dev"],
                    tokenizer=yaml_config["tokenizer"])
    config.init_from_dict(yaml_config)
    
    traingenerator = BatchGenerator(config, "train")
    validgenerator = BatchGenerator(config, "dev")
    model = TransformerLM(config)
    model.to(config.device)
    train(model,
          traingenerator,
          validgenerator,
          config.device,
          config.batch_size,
          config.epochs,
          config.lr)


if __name__ == "__main__":
    main()


