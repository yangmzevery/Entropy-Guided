import os
import torch
import torch.optim as optim
import utils
from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path
from solvers.runners import train, test, load_and_tokenize
from solvers.loss import loss_dict
from time import localtime, strftime
import logging
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, DataCollatorWithPadding
import logging
import csv
import copy


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states, None)

class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        return (hidden_states, None)

def replace_attention_with_identity(model, skip_layers):

    model = copy.deepcopy(model)  
    for i in skip_layers:
        model.bert.encoder.layer[i].attention = DummyAttention()
    return model

def replace_block_with_identity(model, skip_layers):
    model = copy.deepcopy(model)
    for i in skip_layers:
        model.bert.encoder.layer[i] = DummyBlock()
    return model



if __name__ == "__main__":
    
    args = parse_args()

    utils.init_distributed_mode(args)
    
    # Determine rank and GPU from environment variables.
    args.rank = int(os.environ.get("RANK", 0))
    args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    
    # Setup logging only on main process.
    if args.rank == 0:
        current_time = strftime("%d-%b", localtime())
        model_save_path = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args)}"
        if not os.path.isdir(model_save_path):
            mkdir_p(model_save_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(model_save_path, "train.log")),
                logging.StreamHandler()
            ],
            force=True
        )
        logging.info(f"Rank {args.rank}: Logging initialised on main process.")
        logging.info(f"Setting up logging folder : {model_save_path}")
    else:
        logging.basicConfig(level=logging.ERROR)
    
    # For reproducibility:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    
    device = torch.device(args.device)
    
    # Load tokenizer and dataset.
    logging.info("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_datasets = load_and_tokenize(args, tokenizer)
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]
    
    # Distributed training
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        if args.dist_eval:
            if len(val_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            val_sampler = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, collate_fn=data_collator)
    
    # Build a BERT model from scratch using its configuration.
    logging.info("Building model from scratch...")
    config = AutoConfig.from_pretrained(args.model)
    
    # # Determine the number of classes from the dataset:
    # if hasattr(train_dataset.features["label"], "names"):
    #     num_labels = len(train_dataset.features["label"].names)
    # else:
    #     num_labels = len(set(train_dataset["label"].tolist()))
    config.num_labels = args.num_labels
    test_criterion = loss_dict["cross_entropy"]()
    theory_metrics = []
    criterion = loss_dict[args.loss](args, device)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        ignore_mismatched_sizes=True 
    )
 
    ## load checkpoint ##
    checkpoint = torch.load("checkpoint/20_newsgroups/baseline.pth", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    ## calculate entropy ##
    val_loss, val_acc, sce, ece, aece, entropy = test(val_loader, model, test_criterion, 0, theory_metrics, device, loss=args.loss, num_bins=args.num_bins, update_gamma_every=args.update_gamma_every, loss_function=criterion, num_labels=args.num_labels)


    ## replace attention or block with identity ##
    model = replace_block_with_identity(model,[])
    # model = replace_attention_with_identity(model, [])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    if args.rank == 0:
        logging.info(f"Rank {args.rank}: Model built. Number of classes: {args.num_labels}")
    
    # Setup optimizer.
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.loss == 'OLS':
        criterion = loss_dict[args.loss](args.num_labels, device)
    if args.loss == 'AdaFocal':
        criterion = loss_dict[args.loss](args, device)
    else:
        criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta, loss=args.loss)
    
    test_criterion = loss_dict["cross_entropy"]()

    # Calculate total number of train steps
    num_batches_per_epoch = len(tokenized_datasets['train']) // args.batch_size + int(len(tokenized_datasets['train']) % args.batch_size != 0)
    total_train_steps = num_batches_per_epoch * args.epochs

    # Setup learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_train_steps * 0.1,  # 10% warmup
        num_training_steps=total_train_steps
    )

    start_epoch = args.start_epoch
    best_acc = 0.
    best_acc_stats = {}

    # To support theoretical analysis
    theory_metrics = []

    for epoch in range(start_epoch, args.epochs):
        if args.rank == 0:
            logging.info("Epoch [%d/%d] - LR: %.6f", epoch+1, args.epochs, get_lr(optimizer))
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, criterion, device)

        if args.loss == 'OLS':
            criterion.normalise_loss_lams()

        if args.loss == 'AdaFocal':
            val_loss, val_acc, sce, ece, aece, entropy = test(val_loader, model, test_criterion, epoch, theory_metrics, device, loss=args.loss, num_bins=args.num_bins, update_gamma_every=args.update_gamma_every, loss_function=criterion, num_labels=args.num_labels)
        else:
            val_loss, val_acc, sce, ece, aece = test(val_loader, model, test_criterion, epoch, theory_metrics, device)
        
        # scheduler.step()
        
        if args.rank == 0:
            logging.info("Epoch {}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f} | SCE : {:.5f} | ECE : {:.5f} | AECE : {:.5f}".format( 
                         epoch+1, train_loss, train_acc, val_loss, val_acc, sce, ece, aece))
            
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            
            save_checkpoint({
                "epoch": epoch+1,
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "dataset": args.dataset,
                "model": args.model,
            }, is_best, model_save_path)

            if is_best:
                best_acc_stats = {
                    "Accuracy": val_acc,
                    "SCE": sce,
                    "ECE": ece,
                    "AECE": aece
                }
    
    if args.rank == 0:
        logging.info("Training completed...")
        logging.info("The stats for best trained model on test set are as below:")
        logging.info(best_acc_stats)


        # To support theoretical analysis
        fieldnames = theory_metrics[0].keys()
        filename = os.path.join(model_save_path, "theory_metrics.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(theory_metrics)































