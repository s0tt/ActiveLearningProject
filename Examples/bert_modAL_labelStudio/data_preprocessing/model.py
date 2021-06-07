# delay evaluation of annotation
from __future__ import annotations

import os
import re
from typing import Dict, OrderedDict, Tuple, Union

import torch
from torch import Tensor
from transformers import BertModel


class BertQA(torch.nn.Module):
    def __init__(self, cache_dir: Union[None, str] = None):
        super().__init__()
        self.embedder = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.qa_outputs = torch.nn.Linear(self.embedder.config.hidden_size, 2, bias=True) # TODO include bias?
        self.qa_outputs.apply(self.embedder._init_weights)

    def forward(self, token_ids, segment_ids, lengths) -> Tuple[Tensor, Tensor]:
        # input is batch x sequence
        # NOTE the order of the arguments changed from the pytorch pretrained bert package to the transformers package
        embedding, _ = self.embedder(token_ids, token_type_ids=segment_ids, attention_mask=lengths)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)
        # type hints
        start_logits: Tensor
        end_logits: Tensor
        # split last dim to get separate vectors for start and end
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        
        # set padded values in output to small value # TODO consider the same for question + tokens?
        mask = (1-lengths).bool()
        start_logits.masked_fill_(mask, -1e7)
        end_logits.masked_fill_(mask, -1e7)

        return start_logits, end_logits

    # staticmethod since class might be shadowed by DataParallel
    @staticmethod
    def save(dir: str, model: BertQA, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, training_steps: int, value=None, remove_old: bool = True, identifier: str = '', best: bool = False):
        # dir = os.path.join(dir, 'checkpoints')
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        if hasattr(model, 'module'):
            # if model is a DataParallel model then the actual model is wrapped in member 'module'
            model = model.module

        # print("Saving model state dict")
        # print(list(model.state_dict().keys()))
        # exit()
        # print(list(model.state_dict().items())[9])
        params = {
            'identifier': identifier,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_steps': training_steps,
            'value': value
                }
        if scheduler is not None:
            params['scheduler_state_dict'] = scheduler.state_dict()
            
        filename = 'best.pth' if best else f'{training_steps}.pth'
        torch.save(params, os.path.join(dir, filename))

        if remove_old:
            # remove previous checkpoints
            for f in os.listdir(dir):
                match = re.match(r'(\d+)\.pth', f)
                if match and int(match.group(1)) != training_steps:
                    os.remove(os.path.join(dir, f))


    # staticmethod since class might be shadowed by DataParallel
    @staticmethod
    def load(dir: str, model: BertQA = None, device: torch.device = None, best: bool = False) -> Tuple[bool, Union[Dict[str, Tensor], OrderedDict[str, Tensor]], Union[Dict[str, Tensor], OrderedDict[str, Tensor]], int, float, str]:
        # load checkpoint
        if best:
            filename = 'best.pth'
            # check if best model exists
            if not os.path.exists(os.path.join(dir, filename)):
                return False, None, None, 0, None, None
        else:
            if not os.path.exists(dir):
                return False, None, None, 0, None, None
            files = [re.match(r'(\d+)\.pth', f) for f in os.listdir(dir)]
            if not any(files):
                # no checkpoints in path
                return False, None, None, 0, None, None
            checkpoints = [int(checkpoint.group(1)) for checkpoint in files if checkpoint]
            filename = '%s.pth' % max(checkpoints)

        checkpoint = torch.load(os.path.join(dir, filename), map_location=device)

        if model is not None:
            if isinstance(model, torch.nn.DataParallel):
                # if model is a DataParallel model then the actual model is wrapped in member 'module'
                model = model.module

            # load state dict for model
            model.load_state_dict(checkpoint['model_state_dict'])

        # instead of loading the optimizer and scheduler we just return them
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict'] if 'scheduler_state_dict' in checkpoint else None
        training_steps = checkpoint['training_steps']
        value = checkpoint['value']
        identifier = checkpoint['identifier'] if 'identifier' in checkpoint else ''

        # print("Loaded model state dict")
        # print(list(checkpoint['model_state_dict'].items())[9])

        return True, optimizer_state_dict, scheduler_state_dict, training_steps, value, identifier
