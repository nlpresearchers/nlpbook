import argparse
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint, args.config, args.pytorch_dump_path)