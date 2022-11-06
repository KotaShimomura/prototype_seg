from __future__ import absolute_import, division, print_function

from lib.loss.loss_proto import PixelPrototypeCELoss
from lib.utils.distributed import is_distributed
from lib.utils.tools.logger import Logger as Log

SEG_LOSS_DICT = {"pixel_prototype_ce_loss": PixelPrototypeCELoss}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if is_distributed():
            print("use distributed loss")
            return loss

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get("loss", "loss_type") if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error("Loss: {} not valid!".format(key))
            exit(1)
        Log.info("use loss: {}.".format(key))
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)
