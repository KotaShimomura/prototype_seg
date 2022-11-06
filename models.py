from __future__ import absolute_import, division, print_function

from lib.models.nets.hrnet import HRNet_W48_Proto

SEG_MODEL_DICT = {"hrnet_w48_proto": HRNet_W48_Proto}


class Segmentation_Model(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get("network", "model_name")
        # model_name = "hrnet_w48_proto"

        if model_name not in SEG_MODEL_DICT:
            print("your select model not found")
        print(model_name)
        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
