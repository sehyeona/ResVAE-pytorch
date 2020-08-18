import os
import torch

class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)
    
    def save(self, step):
        fname = self.fname_template.format(step)
        print("Saving Checkpoints into %s" %fname)
        outdict = dict()
        for name, module in self.module_dict.items():
            outdict[name] = module
        torch.save(outdict)

    def load(self, step):
        fname = self.fname_template.format(step)
        print(fname)
        assert os.path.exists(fname), fname + " does not exist!"
        print("Loading Checkpoint from %s..." % fname)
        if torch.cuda.is_available() :
            module_dict = torch.load(fname)
        else :
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        # for name, module in self.module_dict.items():
        #     # module = torch.load(module_dict[name])
        #     module.load_state_dict(module_dict[name])

