import logging
from collections import Counter, Sequence
from enum import unique
from functools import partial
from itertools import cycle, takewhile
from typing import Any, Callable, Dict, List, Union

import numpy
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from data_preprocessing.data import (BertQASampler, MRQADataset, SlidingWindowHandler,
                  normalize_answer, pad_batch)
from data_preprocessing.model import BertQA
from data_preprocessing.utils import extract_span, store_metrics


class Metrics():
    """ A class for computing and recording metrics within one epoch."""

    def __init__(self):
        # loss
        self.loss = 0.0
        self.total_loss = 0

        # em + f1
        self.em = 0
        self.f1 = 0.0
        self.total = 0

    def __getitem__(self, key):
        return self.get()[key]

    def get(self):
        return {
            'loss': self.loss/self.total_loss if self.total_loss > 0 else 0.0,
            'em': 100 * self.em/self.total if self.total > 0 else 0.0,
            'f1': 100 * self.f1/self.total if self.total > 0 else 0.0,
            }

    def __repr__(self):
        return self.get()

    def __str__(self):
        return str(self.get()) 

    def add_loss(self, loss, num_samples):
        self.total_loss += num_samples
        self.loss += loss

    @staticmethod
    def em(prediction, truth):
        return int(prediction == truth)

    @staticmethod
    def f1(prediction, truth):
        prediction_tokens = Counter(prediction.split())
        truth_tokens = Counter(truth.split())
        num_overlaps = sum((prediction_tokens & truth_tokens).values())
        if num_overlaps > 0:
            # f1 score is bigger than 0
            precision = num_overlaps / sum(prediction_tokens.values())
            recall = num_overlaps / sum(truth_tokens.values())
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    @staticmethod
    def max_over_ground_truths(metric_fn, prediction, truths):
        return max((metric_fn(prediction, truth) for truth in truths))

    def add(self, predicted_answer, gold_answers: List[str]):
        self.total += 1
        
        # metrics
        # exact match (EM) score
        self.em += Metrics.max_over_ground_truths(Metrics.em, predicted_answer, gold_answers)
        # F1 score
        self.f1 += Metrics.max_over_ground_truths(Metrics.f1, predicted_answer, gold_answers)


def get_score_and_plot(metrics_dict, metric: str = 'f1', writer = None, tag: str = None, step: int = None):
    """Return the average score and plot to tensorboard."""
    # compute average f1 score and plot to tensorboard
    if metric is None:
        score = None
    else:
        score = 0.0
        for metrics in metrics_dict.values():
            score += metrics[metric]
        score /= len(metrics_dict)
        
    if writer:
        if tag is None:
            raise ValueError('tag may not be None')

        # plot eval metrics
        for _data, _metrics in metrics_dict.items():
            for metric, value in _metrics.items():
                writer.add_scalar(f'{tag}-{_data}/{metric}', value, global_step=step)
    return score


def get_per_sample_indices(batch: Dict[str, Any]):
    # NOTE this does expect that chunks of the same question only occur consecutively
    _, indices, counts = numpy.unique(batch['metadata']['qid'], return_index=True, return_counts=True)
    counts = [counts[index] for index in numpy.argsort(indices)]
    per_sample_indices = torch.tensor(list(range(batch['input'].size(0)))).split(counts, dim=0)
    return per_sample_indices


class MRQAAgent():
    def __init__(self, model_dir: str, cache_dir: str, pretrained_model_dir: str = None, disable_cuda: bool = False, results: str = 'results', load_best: bool = False):
        """A class which is able to train on data and evaluate."""
        self.cache_dir = cache_dir

        self.model_dir = model_dir
        self.writer = None
        self.results = results

        # instantiate model # TODO allow for other contextualized embeddings
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        self.model = BertQA(cache_dir=self.cache_dir)
        self.sample_processor = SlidingWindowHandler(self.tokenizer, 512)
        
        # push model to GPU (if wanted) before optimizer is initialized!
        if torch.cuda.is_available() and not disable_cuda:
            # use cuda
            logging.info('Running with CUDA on all available GPUs (%s)', torch.cuda.device_count())
            self.device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                # use all available GPUs
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            # TODO find max batch size for GPUs
        else:
            logging.info('CUDA is not available or use is disabled, running on CPU')
            self.device = torch.device('cpu')

        logging.info('Model dir is %s', self.model_dir)

        # load model
        if pretrained_model_dir is not None:
            # load pre-trained model and reset some variables
            resume, _, _, _, _, _ = BertQA.load(pretrained_model_dir, self.model, self.device, best=True)
            # do not use optimizer, scheduler and training steps if pretrained model has been loaded
            self.optimizer_state_dict, self.scheduler_state_dict, self.training_steps = None, None, 0
            self.best_model_score = None
        
            if not resume:
                logging.error('No pre-trained model found at %s', pretrained_model_dir)
                exit(-1)
            else:
                logging.info('Pre-trained model loaded (from %s) -> resetting training steps, optimizer & scheduler', pretrained_model_dir)
        else:
            resume, self.optimizer_state_dict, self.scheduler_state_dict, self.training_steps, score, _ = BertQA.load(model_dir, self.model, self.device, best=load_best)
            if load_best:
                self.best_model_score = score
            else:
                _, _, _, _, self.best_model_score, _ = BertQA.load(model_dir, best=True)

            if not resume:
                logging.info('No model found (at %s), starting from scratch', model_dir)
            else:
                logging.info('Existing model loaded (from %s)', model_dir)

    def add_tb_writer(self, writer: SummaryWriter, plot_graph: bool = True):
        self.writer = writer
        self.writer.add_graph(self.model, (torch.tensor([[0,1,0]], device=self.device),torch.tensor([[0,0,1]], device=self.device),torch.tensor([[1,1,1]], device=self.device)))
            
    def train(self, data_train: MRQADataset, validation_datasets: Union[None, MRQADataset], data_eval: Union[None, MRQADataset, List[MRQADataset]], batch_size: int, lr: float = 3e-05, num_epochs: int = -1, num_training_steps: int = -1, num_total_training_steps: int = -1, warmup_ratio: float = .1, labels: str = 'single', eval_interval: int = 1000):
        """Perform training, validation and evaluation. (Validation not yet implemented)"""

        def train_step_callback(pbar, batch_sampler):
            """The function used as callback in the training loop."""
            # used as callback before each training step; will update progress bar
            pbar.update()
            pbar.set_description_str(f'Training epoch {batch_sampler.epoch}') # NOTE epoch is increased once new iterator for repetition is created
        
        if num_epochs != -1 and (num_training_steps != -1 or num_total_training_steps != -1):
            raise ValueError('num_epochs is mutually exclusive with num_training_steps and num_total_training_steps')


        if data_eval and not isinstance(data_eval, Sequence):
            data_eval = [data_eval]

        # set up data iterator
        batch_sampler = BertQASampler(data_source=data_train, batch_size=batch_size, training=True, shuffle=True, drop_last=False, fill_last=True, repeat=True)
        batch_sampler_iterator = iter(batch_sampler)
        dataloader = DataLoader(data_train, batch_sampler=batch_sampler_iterator, collate_fn=pad_batch)

        # TODO does it make sense to watch metrics during training?
        train_metrics = Metrics()

        # set up list for training steps
        if num_training_steps > -1 or num_total_training_steps > -1:
            # do at most num_total_training_steps in total
            if num_total_training_steps > -1:
                if num_training_steps > -1:
                    total_steps = max(0, min(num_training_steps, num_total_training_steps - self.training_steps))
                else:
                    total_steps = max(0, num_total_training_steps - self.training_steps)
            else:
                total_steps = num_training_steps
            steps_list = total_steps // eval_interval * [eval_interval]
            if total_steps % eval_interval != 0:
                steps_list.append(total_steps % eval_interval)
        elif num_epochs > -1:
            # steps list contains training steps so that evaluation is performed after each epoch
            steps_list = torch.unique(torch.tensor([epoch for epoch in takewhile(lambda x: x < num_epochs, batch_sampler.get_epochs())]), sorted=True, return_counts=True)[1].tolist()
            total_steps = sum(steps_list)
        else:
            # run infinite training
            total_steps = -1
            steps_list = cycle([eval_interval])

        if total_steps == 0:
            logging.info('Nothing to train.')
            return

        if num_epochs > -1:
            logging.info('Training for %s epoch(s) on %s', num_epochs, data_train)
        if total_steps > -1:
            logging.info('Training for %s step(s) on %s', total_steps, data_train)
        else:
            logging.info('Training infinitely on %s and evaluating every %d training steps', data_train, eval_interval)
        if not validation_datasets:
            logging.info('Validation disabled')
        else:
            logging.info('Validating on %s', ', '.join(map(str, validation_datasets)))
        if not data_eval:
            logging.info('Evaluation disabled')
        else:
            logging.info('Evaluating on %s', ', '.join(map(str, data_eval)))

        # initialize & log train parameters
        logging.info("Learning rate set to %g", lr)
        max_grad_norm = 1.0

        # initialize loss
        if labels == 'single':
            logging.info("Using single label loss")
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1) # ignore out of context index
        else:
            logging.info("Using multi label loss")
            loss_fn = torch.nn.MultiLabelSoftMarginLoss()

        # initialize optimizer
        # TODO do I have to load model.module.parameters() here in case of multiple GPUs?
        optimizer = AdamW(self.model.parameters(), lr=lr, correct_bias=False)
        if self.optimizer_state_dict is not None:
            # load optimizer state
            optimizer.load_state_dict(self.optimizer_state_dict)
            # mark optimizer state dict as consumed
            self.optimizer_state_dict = None

        # initialize scheduler (for lr)
        if total_steps > 0:
            num_warmup_steps = int(total_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
            # TODO parameters might be different!
            if self.scheduler_state_dict is not None:
                # load scheduler state
                scheduler.load_state_dict(self.scheduler_state_dict)
                # mark scheduler state dict as consumed
                self.scheduler_state_dict = None
        else:
            scheduler = None

        # run evaluation before first epoch on train and evaluation datasets
        self.evaluate(data_train, batch_size, loss_fn=loss_fn) # will plot to tensorboard
        if data_eval:
            self.evaluate(data_eval, batch_size, loss_fn=loss_fn) # will plot to tensorboard
        
        # loop over steps
        with tqdm(total=total_steps, smoothing=0) as pbar:
            data_iter = iter(dataloader) # create iterator so that the same can be used in all function calls (also working with zip)
            # pbar.update() # somehow we need this here

            for steps in steps_list:
                self.train_steps(data_iter, loss_fn, optimizer, scheduler, steps, train_metrics, callback_fn=partial(train_step_callback, pbar, batch_sampler_iterator))
                # TODO add patience
                # plot train metrics
                loss = get_score_and_plot(self.evaluate(data_train, batch_size, loss_fn=loss_fn), 'loss', None)
                # save model checkpoint
                BertQA.save(self.model_dir, self.model, optimizer, scheduler, self.training_steps, loss, identifier=data_train.identifier)
                
                # evaluate
                if data_eval:
                    # save as best model only if performance is better
                    score = get_score_and_plot(self.evaluate(data_eval, batch_size, loss_fn=loss_fn), 'f1', None)

                    if self.best_model_score is None or score > self.best_model_score:
                        logging.info('Saving model since performance increased')
                        BertQA.save(self.model_dir, self.model, optimizer, scheduler, self.training_steps, score, identifier=data_train.identifier, best=True)
                        self.best_model_score = score

    def train_steps(self, data_iter, loss_fn, optimizer, scheduler, train_steps: int, metrics, callback_fn: Callable = None):
        """Run training for `train_steps` steps."""

        for self.training_steps, batch in zip(range(self.training_steps + 1, self.training_steps + 1 + train_steps), data_iter):
            # perform one training step each time (with batches)
            if callback_fn:
                callback_fn()
            self.model.train()
            self.step(batch, loss_fn, metrics, training=True, evaluation=False, optimizer=optimizer, scheduler=scheduler)
        
    def step(self, batch, loss_fn: Callable, metrics: Metrics, training: bool, evaluation: bool, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler.LambdaLR = None):
        """Perform one step of training or evaluation."""

        if training and optimizer is None:
            raise ValueError('training can be set to True only if loss_fn and optimizer have been provided')

        # model prediction
        start_logits, end_logits = self.model(batch['input'].to(self.device), batch['segments'].to(self.device), batch['mask'].to(self.device))
        
        if loss_fn is not None:
            # extract label
            # NOTE label contains offset for question + tokens
            if isinstance(loss_fn, (torch.nn.MultiLabelMarginLoss, torch.nn.MultiLabelSoftMarginLoss)):
                label_start, label_end = batch['label_multi'].to(self.device).split(1, dim=1) # all spans for this sample
            else:
                label_start, label_end = batch['label'].to(self.device).split(1, dim=1) # one span

            # compute loss
            start_loss = loss_fn(start_logits, label_start.squeeze(1))
            end_loss = loss_fn(end_logits, label_end.squeeze(1))
            loss = start_loss + end_loss # TODO mean instead of sum?
            metrics.add_loss(loss.detach().cpu().item(), 1) # 1 since loss is already mean over batch

            if training:
                # perform weight update
                optimizer.zero_grad()
                loss.backward()
                # TODO gradient clipping not applied yet
                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        if evaluation:
            # extract answer and compute metric

            predicted_answers = self.predict_answer(batch, start_logits, end_logits) # returns one answer for each question
            per_sample_indices =  get_per_sample_indices(batch)
            
            # per question prediction (considering all precictions)
            for predicted_answer, sample_indices in zip(predicted_answers, per_sample_indices):
                # here we can use any index of the current indices since they all belong to the same sample
                idx = sample_indices[0].item()

                # since we use the list of answers we don't have to use the answers of the best span since they are all the same
                true_answers = {normalize_answer(answer) for answer in batch['metadata']['answers_per_instance'][idx]}
                paragraph = batch['metadata']['paragraph'][idx]
                context = batch['metadata']['context'][idx]
                wordpiece_to_token_idx = batch['metadata']['wordpiece_to_token_idx'][idx]
                token_to_context_idx = batch['metadata']['token_to_context_idx'][idx]

                # NOTE fix for some differing answers: add extracted answer from true span
                answers = batch['metadata']['original_answers'][idx]
                for _, span in answers:
                    extracted_answer = normalize_answer(context[token_to_context_idx[wordpiece_to_token_idx[span[0]]][0]:token_to_context_idx[wordpiece_to_token_idx[span[1]]][1]])
                    # print("Extracted answer from original span:", extracted_answer)
                    true_answers.add(extracted_answer)
                # print("True answers used for evaluation:", true_answers)

                # compute metrics
                metrics.add(predicted_answer, true_answers)


    def predict_answer(self, batch, start_logits: Tensor = None, end_logits: Tensor = None, extract_true_answer: bool = False):
        """ Predict the answer for the given batch. If given, the start_logits and end_logits should match batch indices. Returns one answer per question id."""

        if start_logits is None or end_logits is None:
            # model prediction
            start_logits, end_logits = self.model(batch['input'].to(self.device), batch['segments'].to(self.device), batch['mask'].to(self.device))
        
        # predict span scores
        span_scores, spans, answers = extract_span(start_logits.detach().cpu(), end_logits.detach().cpu(), batch)
        
        per_sample_indices =  get_per_sample_indices(batch)
        
        answers = []
        # per question prediction (considering all precictions)
        for sample_indices in per_sample_indices:
            best_span_idx = torch.tensor([span_scores[idx] for idx in sample_indices]).argmax()
            # use best_span_idx_batch for further access
            best_span_idx_batch = sample_indices[best_span_idx].item()

            # since we use the list of answers we don't have to use the answers of the best span since they are all the same
            context = batch['metadata']['context'][best_span_idx_batch]
            wordpiece_to_token_idx = batch['metadata']['wordpiece_to_token_idx'][best_span_idx_batch]
            token_to_context_idx = batch['metadata']['token_to_context_idx'][best_span_idx_batch]

            # extract the answer from the predicted span
            predicted_span = spans[best_span_idx_batch]
            answer_span_offset = batch['metadata']['context_instance_start_end'][best_span_idx_batch][0] - len(batch['metadata']['question_tokens'][best_span_idx_batch]) - 2
            predicted_span = (predicted_span[0] + answer_span_offset, predicted_span[1] + answer_span_offset)
            prediction_start_token, prediction_end_token = (wordpiece_to_token_idx[_idx.item()] for _idx in predicted_span)
            prediction_start_idx_no_offset, prediction_end_idx_no_offset = token_to_context_idx[prediction_start_token][0], token_to_context_idx[prediction_end_token][1]
            answers.append(normalize_answer(context[prediction_start_idx_no_offset:prediction_end_idx_no_offset]))

        return answers
        
    def evaluate(self, data: Union[List[MRQADataset], MRQADataset], batch_size: int, loss_fn = None):
        """Evaluate given data once."""

        if not isinstance(data, (list, tuple, set)):
            data = [data]

        metrics_dict = {}
        self.model.eval()
        for _data in data:
            metrics = Metrics()
            # data
            batch_sampler = BertQASampler(data_source=_data, batch_size=batch_size, training=False, shuffle=False)
            dataloader = DataLoader(_data, batch_sampler=batch_sampler, collate_fn=pad_batch)

            # process whole data
            with torch.no_grad():
                with tqdm(total=len(_data), smoothing=0) as pbar:
                    pbar.set_description_str(f"Evaluating training step {self.training_steps} on {_data.identifier}")
                    for batch in dataloader:
                        # save number of processed samples before calling step() in order to update progress bar with correct value
                        num_samples_processed = metrics.total
                        self.step(batch, loss_fn, metrics, training=False, evaluation=True)
                        # progess bar keeps track of samples
                        num_samples_current_batch = metrics.total - num_samples_processed
                        pbar.update(n=num_samples_current_batch)

                metrics_dict[_data.identifier] = metrics.get()
            
            if self.writer:
                for metric, value in metrics.get().items():
                    self.writer.add_scalar(f'eval-{_data.identifier}/{metric}', value, global_step=self.training_steps)

        return metrics_dict
