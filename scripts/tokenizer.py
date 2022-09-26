from argparse import ArgumentParser
import sentencepiece as spm

def train(input_text: str, vocab_size: int=12_000):
    """Train a bpe tokenizer model."""
    spm.SentencePieceTrainer.train(input=input_text,
                                    model_prefix='fula',
                                    vocab_size=vocab_size,
                                    model_type="bpe",
                                    pad_id=3,
                                    character_coverage=1)

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_text",
                        help="The text on which the tokenizer will be trained.",
                        required=True)
    parser.add_argument("-v", "--vocab_size",
                        help="The size of the vocabulary.",
                        default=12_000,
                        required=False)
    args = parser.parse_args()
    train(args.input_text, args.vocab_size)

if __name__ == "__main__":
    main()

                            
