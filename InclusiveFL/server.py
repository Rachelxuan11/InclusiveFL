import torch
import logging
import math
import re
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_scheduler,
)

import sys
sys.path.append('../src/glue/')


class Aggregator:
    def __init__(self, i, args, num_labels, num_train_sam):
        # config
        self.config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                            num_labels=num_labels, finetuning_task=args.task_name,
                                            num_hidden_layers= args.num_hidden_layers[i])

        # model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
        )
        model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
        logging.info("Num of parameters for model {} = {}".format(i, model_size))

        # optim
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate[i])
        # self.optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate)

        # lr scheduler
        num_update_steps_per_round = math.ceil(num_train_sam * args.sample_ratio)

        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_update_steps_per_round * args.rounds,
        )

        self._init_grad_param()
    
    def _init_grad_param(self):
        self.optimizer.zero_grad()
    
    def update(self):
        self.update_model_grad()
        self._init_grad_param()
    
    def update_model_grad(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
    
    def collect(self, model_grad):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = model_grad[name]
            else:
                param.grad += model_grad[name]


class Aggregator_mom(Aggregator):
    def __init__(self, i, args, num_labels, num_train_sam):
        self.args = args
        self.idx = i
        # config
        self.config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                            num_labels=num_labels, finetuning_task=args.task_name,
                                            num_hidden_layers= args.num_hidden_layers[i])

        # model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
        )
        model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
        logging.info("Num of parameters for model {} = {}".format(i, model_size))

        # optim
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate[i])
        # self.optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate)

        self.param_name = set(n for n, p in self.model.named_parameters() if p.requires_grad)
        self.grad_cache = [{} for _ in range(args.num_hidden_layers[i])]
        self.grad_momentum = [{} for _ in range(args.num_hidden_layers[i])]
        for layer_id in range(args.num_hidden_layers[i]):
            # iter over all param names and collect param names for layer_id
            for pn in self.param_name:
                if f'layer.{layer_id}.' in pn:
                    rename_pn = get_rename_shortcut(pn)
                    self.grad_cache[layer_id][rename_pn] = 0
                    self.grad_momentum[layer_id][rename_pn] = 0

        # lr scheduler
        num_update_steps_per_round = math.ceil(num_train_sam * args.sample_ratio)

        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_update_steps_per_round * args.rounds,
        )

        self._init_grad_param()

    def _init_grad_param(self):
        self.optimizer.zero_grad()
    
    def update(self):
        self.keep_grad()
        self.load_grad_mem()
        self.update_model_grad()
        self._init_grad_param()

    def load_grad_mem(self):
        pattern = re.compile(f'layer.(\d+).')
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'layer.' in name:
                rename, layer_id = get_rename_shortcut(name, True)
                if f'layer.{self.args.num_hidden_layers[self.idx]-1}' in name and isinstance(self.grad_momentum[layer_id][rename], int):
                    param.grad = self.grad_momentum[layer_id][rename] * self.args.mom_beta + (1-self.args.mom_beta) * param.grad

    def update_model_grad(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
    
    def collect(self, model_grad):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = model_grad[name]
            else:
                param.grad += model_grad[name]

    def keep_grad(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'layer' in name:
                rename, layer_id = get_rename_shortcut(name, True)
                self.grad_cache[layer_id][rename] = param.grad


def HeteAgg(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, did2_sample_portion):
    # print("!!!!!!!!! HETE AGG !!!!!!!!!!! (share sub-large model)")
    with torch.no_grad():
        avg_params = {}

        # collect local parameters
        for name in dummy_names[-1]:
            if name in hete_exclude_name:
                continue
            # mask = torch.LongTensor([name in x for i, x in enumerate(dummy_names) if name not in leave_one_names[i] else False])
            mask = []
            for i, x in enumerate(dummy_names):
                if name in x and not(name in leave_one_names[i]) and not(i in args.drop_idx):
                    mask.append(1)
                else:
                    mask.append(0)
            mask = torch.LongTensor(mask)
            if sum(mask) != 0:
                avg_params[name] = 0
            else:
                continue

            sam_weight = torch.FloatTensor(did2_sample_portion) * mask
            sam_weight = sam_weight.cuda()

            sam_weight = sam_weight / torch.sum(sam_weight)

            for i in range(args.num_hete_total):
                if mask[i] == 0:
                    continue
                for n, p in aggs[i].model.named_parameters():
                    if n == name:
                        try:
                            avg_params[name] += p * sam_weight[i]
                        except:
                            import pdb; pdb.set_trace()
                        # if r == 0:
                        #     logger.info(f"Hete-agg {name} from user {i} with sample weight {sam_weight[i]}")
                        break

        # apply global parameters
        for i in range(args.num_hete_total):
            if i in args.drop_idx:
                continue
            for name, param in aggs[i].model.named_parameters():
                if name in hete_exclude_name or name in leave_one_names[i]:
                    continue
                if name in avg_params:
                    param.copy_(avg_params[name])

        del avg_params

def HeteAgg_mom(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, did2_sample_portion):
    # print("!!!!!!!!! HETE MOM AGG !!!!!!!!!!! (share sub-large model)")
    with torch.no_grad():
        avg_params = {}
        sam_weight_dict = {}

        # collect local parameters
        for name in dummy_names[-1]:
            if name in hete_exclude_name:
                continue
            
            mask = []
            for i, x in enumerate(dummy_names):
                if name in x and not(name in leave_one_names[i]) and not(i in args.drop_idx):
                    mask.append(1)
                else:
                    mask.append(0)
            mask = torch.LongTensor(mask)
            if sum(mask) != 0:
                avg_params[name] = 0
            else:
                continue

            sam_weight = torch.FloatTensor(did2_sample_portion) * mask
            sam_weight = sam_weight.cuda()

            sam_weight = sam_weight / torch.sum(sam_weight)
            sam_weight_dict[name] = sam_weight

            if name in hete_exclude_name:
                sam_weight_dict[name] = sam_weight
                continue

            for i in range(args.num_hete_total):
                if mask[i] == 0:
                    continue
                for n, p in aggs[i].model.named_parameters():
                    if n == name:
                        avg_params[name] += p * sam_weight[i]
                        # if r == 0:
                        #     logger.info(f"Hete-agg {name} from user {i} with sample weight {sam_weight[i]}")
                        break

        # apply global parameters
        for i in range(args.num_hete_total):
            if i in args.drop_idx:
                continue
            for name, param in aggs[i].model.named_parameters():
                if name in hete_exclude_name or name in leave_one_names[i]:
                    continue
                if name in avg_params:
                    param.copy_(avg_params[name])

        del avg_params

        # hete agg grad
        assert args.local_one
        for i in range(args.num_hete_total-1):
            if i in args.drop_idx:
                continue
            start_id = args.num_hidden_layers[i]-1
            end_id = args.num_hidden_layers[i+1]-1
            layer_weight = 1.0 / (end_id - start_id + 1)

            for j in range(start_id, end_id+1):
                for layer_rename, layer_grad in aggs[i+1].grad_cache[j].items():
                    aggs[i].grad_momentum[-1][layer_rename] = layer_weight * layer_grad


def evaluate(i, r, args, eval_model, accelerator, eval_dataloader, metric, is_regression, logger, best_eval, best_eval_r, loss, processed_datasets, data_collator):
    eval_model.eval()
    for step, batch in enumerate(eval_dataloader):
        outputs = eval_model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"round {r}: {eval_metric}")

    if args.save_per_round:
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(eval_model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        eval_model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = eval_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

def get_layer_id(name):
    if 'layer' not in name:
        return -1
    pattern = re.compile(f'layer.(\d+).')
    layer_id = int(pattern.findall(name)[0].split('.')[-1])
    return layer_id


def process_model_grad(model_param, num_users_homo):
    model_grad = {}
    for name, param in model_param:
        if param.grad is not None:
            model_grad[name] = param.grad * (1.0 / num_users_homo)
        else:
            model_grad[name] = None
    return model_grad

def get_rename_shortcut(name, return_id=False):
    pattern = re.compile(f'layer.(\d+).')
    layer_id = int(pattern.findall(name)[0].split('.')[-1])
    pn_tmp = name.split('.')
    rename = '.'.join(pn_tmp[pn_tmp.index(str(layer_id)) + 1: ])
    if return_id:
        return rename, layer_id
    else:
        return rename