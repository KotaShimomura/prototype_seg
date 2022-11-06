from __future__ import absolute_import, division, print_function

import argparse
import os

import torch.distributed as dist

from train_utils import Trainer
from utils import Configer

dist.init_process_group(
    "gloo", init_method="file:///tmp/somefile", rank=0, world_size=1
)

DATA_DIR = "/workspace/workspace/ProtoSeg_local/data/Cityscapes"
SAVE_DIR = "/workspace/workspace/ProtoSeg_local/output/Cityscapes/seg_results/"
CHECKPOINTS_ROOT = "/workspace/workspace/ProtoSeg_local/output/Cityscapes"


class CFG:
    model_method = "fcn"
    phase = "train"


def main():
    def str2bool(v):
        """Usage:
        parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                            dest='pretrained', help='Whether to use pretrained models.')
        """
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser.add_argument("--phase", default="train", type=str, dest="phase")
    parser.add_argument(
        "--model_name", default="hrnet_w48_proto", type=str, dest="model_name"
    )
    parser.add_argument(
        "--configs",
        default="configs/cityscapes/H_48_D_4_proto.json",
        type=str,
        dest="configs",
        help="The file of the hyper parameters.",
    )

    parser.add_argument(
        "--gpu", default=[0], nargs="+", type=int, dest="gpu", help="The gpu list used."
    )
    parser.add_argument(
        "--data_dir",
        default="/workspace/workspace/ProtoSeg_local/data/Cityscapes",
        type=str,
        nargs="+",
        dest="data:data_dir",
        help="The Directory of the data.",
    )
    parser.add_argument(
        "--backbone",
        default="hrnet48",
        type=str,
        dest="network:backbone",
        help="The base network of model.",
    )
    parser.add_argument(
        "--checkpoints_name",
        default="hrnet_w48_proto_lr1x_hrnet_proto_80k",
        type=str,
        dest="checkpoints:checkpoints_name",
        help="The name of checkpoint model.",
    )
    parser.add_argument(
        "--loss_type",
        default="pixel_prototype_ce_loss",
        type=str,
        dest="loss:loss_type",
        help="The loss type of the network.",
    )
    parser.add_argument(
        "--drop_last",
        type=str2bool,
        nargs="?",
        default=False,
        dest="data:drop_last",
        help="Fix bug for syncbn.",
    )

    # train config
    parser.add_argument(
        "--gathered",
        type=str2bool,
        nargs="?",
        default=True,
        dest="network:gathered",
        help="Whether to gather the output of model.",
    )
    parser.add_argument(
        "--loss_balance",
        type=str2bool,
        nargs="?",
        default=False,
        dest="network:loss_balance",
        help="Whether to balance GPU usage.",
    )
    parser.add_argument(
        "--log_to_file",
        type=str2bool,
        nargs="?",
        default=True,
        dest="logging:log_to_file",
        help="Whether to write logging into files.",
    )
    parser.add_argument(
        "--max_iters",
        default=80000,
        type=int,
        dest="solver:max_iters",
        help="The max iters of training.",
    )
    parser.add_argument(
        "--checkpoints_root",
        default="/workspace/workspace/ProtoSeg_local/output/Cityscapes",
        type=str,
        dest="checkpoints:checkpoints_root",
        help="The root dir of model save path.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/workspace/workspace/ProtoSeg_local/checkpoints/cityscapes/hrnetv2_w48_imagenet_pretrained.pth",
        dest="network:pretrained",
        help="The path to pretrained model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        dest="train:batch_size",
        help="The batch size of training.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        dest="distributed",
        help="Use multi-processing training.",
    )
    parser.add_argument(
        "--base_lr",
        default=0.01,
        type=float,
        dest="lr:base_lr",
        help="The learning rate.",
    )
    parser.add_argument(
        "--nbb_mult",
        default=1.0,
        type=float,
        dest="lr:nbb_mult",
        help="The not backbone mult ratio of learning rate.",
    )

    # test config
    parser.add_argument(
        "--network", default="hrnet_w48_proto", type=str, dest="network"
    )
    parser.add_argument("REMAIN", nargs="*")
    parser.add_argument(
        "--test_img",
        default=None,
        type=str,
        dest="test:test_img",
        help="The test path of image.",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        dest="test:test_dir",
        help="The test directory of images.",
    )
    parser.add_argument(
        "--out_dir",
        default="none",
        type=str,
        dest="test:out_dir",
        help="The test out directory of images.",
    )
    parser.add_argument(
        "--save_prob",
        type=str2bool,
        nargs="?",
        default=False,
        dest="test:save_prob",
        help="Save the logits map during testing.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        dest="network:resume",
        help="The path of checkpoints.",
    )

    args_parser = parser.parse_args()
    print("return parser")

    return args_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train & test config
    if CFG.model_method == "fcn":
        if CFG.phase == "infer":
            print("Load Model Now ...")
            configer = Configer(args_parser=parser.parse_args())
            data_dir = configer.get("data", "data_dir")

            if isinstance(data_dir, str):
                data_dir = [data_dir]
            abs_data_dir = [os.path.expanduser(x) for x in data_dir]
            configer.update(["data", "data_dir"], abs_data_dir)

            #             project_dir = os.path.dirname(os.path.realpath(__file__))
            #             configer.add(['project_dir'], project_dir)
            model = None

            # model = Tester(configer)
            print("Load Model Finish")

            print("Inference Starting ...")
            args = inference_fn()
            cityscapes_evaluator = CityscapesEvaluator()
            cityscapes_evaluator.evaluate(pred_dir=args.pred_dir, gt_dir=args.gt_dir)

        else:
            print("Train mode")
            print("Load Model Now ...")
            configer = Configer(args_parser=main())
            data_dir = configer.get("data", "data_dir")

            if isinstance(data_dir, str):
                data_dir = [data_dir]
            abs_data_dir = [os.path.expanduser(x) for x in data_dir]
            configer.update(["data", "data_dir"], abs_data_dir)

            #             project_dir = os.path.dirname(os.path.realpath(__file__))
            #             configer.add(['project_dir'], project_dir)
            model = None

            model = Trainer(configer)
            print("create model pass")

            print("start training model")
            model.train()
    else:
        print("attention models will be available")
