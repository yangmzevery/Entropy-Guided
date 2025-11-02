import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy
import numpy as np
from calibration_library.metrics import ECELoss, SCELoss, AdaptiveECELoss
from datasets import load_dataset, Dataset
from sklearn.datasets import fetch_20newsgroups


# AdaFocal
from calibration_library.metrics import adaECE_error_mukhoti as adaECE_error
from calibration_library.metrics import test_classification_net_adafocal


# To support theoretical analysis
from utils import compute_entropy, compute_kl_divergence, compute_cosine_similarity

def load_and_tokenize(args, tokenizer):

    # Determine text column based on dataset
    if args.dataset == "20_newsgroups":
        text_column = "text"
        dataset_name = "SetFit/20_newsgroups"
    elif args.dataset == "ag_news":
        text_column = "text"
        dataset_name = "fancyzhx/ag_news"
    elif args.dataset == "dbpedia":
        text_column = "content"
        dataset_name = "fancyzhx/dbpedia_14"
    elif args.dataset == "imdb":
        dataset_name = "imdb"
        text_column = "text"
    elif args.dataset == "tweet":
        text_column = "text"
    elif args.dataset == "yelp":
        text_column = "text"
    else:
        text_column = "text"  # Default
    
    # # Load dataset from HuggingFace datasets.
    dataset = load_dataset(dataset_name)

    # if args.dataset == "20_newsgroups":
    #     # Fetch the data using scikit-learn
    #     newsgroups = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'))
    #     # Prepare a dictionary
    #     data_dict = {
    #         "text": newsgroups.data,
    #         "label": np.array(newsgroups.target)
    #     }
    #     dataset = Dataset.from_dict(data_dict)
    #     dataset = dataset.train_test_split(test_size=0.2)

    # elif args.dataset == "dbpedia":
    #     dataset = load_dataset(
    #         "csv",
    #         data_files={
    #             "train": "dbpedia_csv/train.csv",
    #             "test": "dbpedia_csv/test.csv"
    #         },
    #         column_names=["label", "title", "content"]
    #     )
    # elif args.dataset == "yelp":
    #     dataset = load_dataset(
    #         "csv",
    #         data_files={
    #             "train": "yelp_review_full_csv/train.csv",
    #             "test": "yelp_review_full_csv/test.csv"
    #         },
    #         column_names=["label", "text"]
    #     )
    #     # Shift labels on all splits
    #     for split in dataset:
    #         dataset[split] = dataset[split].map(lambda x: {"label": x["label"] - 1})
    # elif args.dataset == "tweet":
    #     dataset = load_dataset("tweet_eval", "emotion")
    # elif args.dataset == "emotion":
    #     dataset = load_dataset("dair-ai/emotion", "split")
    # else:
    #     dataset = load_dataset(dataset_name)


    # Tokenize using the provided tokenizer.
    def tokenize_fn(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=args.max_length)
    
    tokenized_datasets = dataset.map(tokenize_fn)
    # Keep only necessary columns (input_ids, attention_mask, label)
    keep_cols = ["input_ids", "attention_mask", "label"]
    tokenized_datasets = tokenized_datasets.remove_columns([col for col in tokenized_datasets["train"].column_names if col not in keep_cols])

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    return tokenized_datasets


def reduce_average_meter(meter):
    """
    Reduce an AverageMeter across all processes.
    Assumes meter has 'sum' and 'count' attributes.
    """
    tensor = torch.tensor([meter.sum, meter.count], device='cuda')
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # Compute the average from the reduced sum and count
    return tensor[0].item() / tensor[1].item()


def train(trainloader, model, optimizer, scheduler, criterion, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    
    for batch_idx, batch in bar:
       
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # compute output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        top1, = accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), input_ids.size(0))
        accs.update(top1.item(), input_ids.size(0))

        # compute gradient and do optimiser and scheduler step
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the scheduler every batch
        optimizer.zero_grad()

        # plot progress locally (each process will print its own progress)
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | acc: {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            loss=losses.avg,
            acc=accs.avg
        ))

    # For distributed training, reduce the metrics from all processes.
    if torch.distributed.is_initialized():
        global_loss = reduce_average_meter(losses)
        global_top1 = reduce_average_meter(accs)
    else:
        global_loss, global_top1 = losses.avg, accs.avg

    return (global_loss, global_top1)

@torch.no_grad()
def test(testloader, model, criterion, epoch, theory_metrics, device, **kwargs):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    all_targets = []
    all_outputs = []

    theory_probs = []
    theory_targets = []


    bar = tqdm(enumerate(testloader), total=len(testloader))
    entropy_all = 0 
    for batch_idx, batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # compute output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        bert_model = model.module if hasattr(model, "module") else model
        encoder_outputs = bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state
        entropy = torch.sum(torch.std(last_hidden_state,dim=(0,1)))
        entropy_all += entropy.item()
        loss = criterion(outputs, labels)
        
        # measure accuracy and record loss
        top1, = accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), input_ids.size(0))
        accs.update(top1.item(), input_ids.size(0))

        # Convert to CPU numpy arrays for gathering later
        all_targets.append(labels.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())

        theory_probs.append(outputs.cpu())
        theory_targets.append(labels.cpu())

        # plot progress locally (each process will print its own progress)
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | acc: {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            loss=losses.avg,
            acc=accs.avg
        ))


    # Concatenate local predictions
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)


    # Gather predictions and targets from all processes if in distributed mode.
    if torch.distributed.is_initialized():
        # Use all_gather_object to collect lists from all ranks
        gathered_outputs = [None for _ in range(torch.distributed.get_world_size())]
        gathered_targets = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_outputs, all_outputs)
        torch.distributed.all_gather_object(gathered_targets, all_targets)
        all_outputs = np.concatenate(gathered_outputs, axis=0)
        all_targets = np.concatenate(gathered_targets, axis=0)

        # Also reduce the scalar metrics
        global_loss = reduce_average_meter(losses)
        global_acc = reduce_average_meter(accs)
    else:
        global_loss, global_acc = losses.avg, accs.avg

    # Compute calibration metrics on the global outputs and targets
    ece = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    sce = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    aece = AdaptiveECELoss().forward(all_outputs, all_targets)


    ######### To support theoretical analysis #########
    
    theory_probs = torch.cat(theory_probs, dim=0)      # [N_val, C]
    theory_targets = torch.cat(theory_targets, dim=0)  # [N_val]
    N, C = theory_probs.shape

    # Batch-level average Q_j for each class
    batch_avg = torch.zeros_like(theory_probs)      # [N, C]
    for j in range(C):
        mask = (theory_targets == j)
        if mask.sum() > 0:
            Qj = theory_probs[mask].mean(dim=0)
            batch_avg[mask] = Qj.unsqueeze(0).expand(mask.sum(), -1)

    # KL divergence KL(Q_j || p_i)
    kl_divs = compute_kl_divergence(theory_probs, batch_avg)
    avg_kl_div = kl_divs.mean().item()

    # Cosine similarity between p_i and Q_j
    cos_sims = compute_cosine_similarity(theory_probs, batch_avg)
    avg_cos_sim = cos_sims.mean().item()

    theory_metrics.append({
        'epoch': epoch,
        'avg_kl_divergence': avg_kl_div,
        'avg_cosine_similarity': avg_cos_sim
        })



    ######### AdaFocal #########
   
    # This updates the Adafocal's gamma-parameter either after every epoch or after a specified number of batches.
    if loss == "AdaFocal" and update_gamma_every == -1 and batch_idx == len(testloader)-1:
        _, _, _, labels, predictions, confidences, _ = test_classification_net_adafocal(model, testloader, device, num_bins=num_bins, num_labels=num_labels)
        ece, bin_dict = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        loss_function.update_bin_stats(val_adabin_dict)
    elif loss == "AdaFocal" and update_gamma_every > 0 and batch_idx > 0 and batch_idx % update_gamma_every == 0:
        _, _, _, labels, predictions, confidences, _ = test_classification_net_adafocal(model, testloader, device, num_bins=num_bins, num_labels=num_labels)
        ece, bin_dict = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        loss_function.update_bin_stats(val_adabin_dict)



    return (global_loss, global_acc, sce, ece, aece, entropy_all)



