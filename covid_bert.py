from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, \
    get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch

RANDOM_SEED = 901
import random

random.seed(RANDOM_SEED)

import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import os
from tqdm import tqdm
import argparse
import time
import datetime

from utils import log_list, make_dir_if_not_exists, load_from_pickle, save_in_json

Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"
RANDOM_SEED = 901
torch.manual_seed(RANDOM_SEED)
POSSIBLE_BATCH_SIZE = 8

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class MultiLabelBertForCovidEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # We will create a dictionary of classifiers based on the number of subtasks
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            entity_start_positions,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # NOTE: outputs[0] has all the hidden dimensions for the entire sequence
        # We will extract the embeddings indexed with entity_start_positions
        pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class COVID19TaskDataset(Dataset):
    """COVID19TaskDataset is a generic dataset class which will read data related to different questions"""

    def __init__(self, instances):
        super(COVID19TaskDataset, self).__init__()
        self.instances = instances
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples


class TokenizeCollator():
    def __init__(self, tokenizer, entity_start_token_id):
        self.tokenizer = tokenizer
        self.entity_start_token_id = entity_start_token_id

    def fix_user_mentions_in_tokenized_tweet(self, tokenized_tweet):
        return ' '.join(["@USER" if word.startswith("@") else word for word in tokenized_tweet.split()])

    def __call__(self, batch):
        all_bert_model_input_texts = list()
        gold_labels = list()
        # text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
        for chunk, tokenized_tweet_with_masked_chunk, label, _ in batch:
            tokenized_tweet_with_masked_chunk = self.fix_user_mentions_in_tokenized_tweet(
                tokenized_tweet_with_masked_chunk)
            if chunk in ["AUTHOR OF THE TWEET", "NEAR AUTHOR OF THE TWEET"]:
                # First element of the text will be considered as AUTHOR OF THE TWEET or NEAR AUTHOR OF THE TWEET
                bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> </E>")

            else:
                bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> " + chunk + " </E>")
            all_bert_model_input_texts.append(bert_model_input_text)
            # Add subtask labels in the gold_labels dictionary
            gold_labels.append(label)
        # Tokenize
        all_bert_model_inputs_tokenized = self.tokenizer(all_bert_model_input_texts, pad_to_max_length=True,
                                                         return_tensors="pt")
        input_ids, token_type_ids, attention_mask = all_bert_model_inputs_tokenized['input_ids'], \
                                                    all_bert_model_inputs_tokenized['token_type_ids'], \
                                                    all_bert_model_inputs_tokenized['attention_mask']

        # First extract the indices of <E> token in each sentence and save it in the batch
        entity_start_positions = (input_ids == self.entity_start_token_id).nonzero()
        # Also extract the gold labels
        labels = torch.Tensor(gold_labels)
        # print(len(batch))
        if entity_start_positions.size(0) == 0:
            # Send entity_start_positions to [CLS]'s position i.e. 0
            entity_start_positions = torch.zeros(input_ids.size(0), 2).long()

        # Verify that the number of labels for each subtask is equal to the number of instances
        if input_ids.size(0) != labels.size(0):
            print('error Bad batch')
            exit()
        assert input_ids.size(0) == labels.size(0)
        return {"input_ids": input_ids, "entity_start_positions": entity_start_positions, "gold_labels": labels,
                "batch_data": batch, "token_type_ids": token_type_ids, "attention_mask": attention_mask}


def make_predictions_on_dataset(dataloader, model, device, dataset_name, dev_flag=False):
    # Create tqdm progressbar
    if dev_flag:
        pbar = dataloader
    else:

        pbar = tqdm(dataloader)
    # Setting model to eval for predictions
    # NOTE: assuming that model is already in the given device
    model.eval()

    all_predictions = list()
    all_prediction_scores = list()
    all_labels = list()

    with torch.no_grad():
        for step, batch in enumerate(pbar):
            # Create testing instance for model
            input_dict = {"input_ids": batch["input_ids"].to(device),
                          "entity_start_positions": batch["entity_start_positions"].to(device),
                          "token_type_ids": batch["token_type_ids"].to(device),
                          "attention_mask": batch["attention_mask"].to(device)
                          }
            labels = batch["gold_labels"].cpu().tolist()
            logits = model(**input_dict)[0]

            predicted_probs = torch.sigmoid(logits)
            predicted_labels = (predicted_probs > 0.5).type(torch.Tensor)

            prediction_scores = predicted_probs.cpu().tolist()
            predicted_labels = predicted_labels.cpu().tolist()

            all_predictions.extend(predicted_labels)
            all_prediction_scores.extend(prediction_scores)
            all_labels.extend(labels)

    return all_predictions, all_labels, all_prediction_scores


def generate_instances(file_name):
    task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(file_name)

    label_dict = dict()

    for idx, keys_tags_tuple in enumerate(question_keys_and_tags):
        label_dict[keys_tags_tuple[0]] = idx

    num_label = len(label_dict)

    instances_dict = dict()  # store instance infor

    for question_tag, items in task_instances_dict.items():
        for item in items:
            new_key = (item[1], item[6], item[0])
            if new_key not in instances_dict.keys():
                instances_dict.setdefault(new_key, [0] * num_label)
            if item[8] == 1:
                instances_dict[new_key][label_dict[question_tag]] = 1

    instances_list = []
    for key, label in instances_dict.items():
        instances_list.append([key[0], key[1], label, key[2]])

    return instances_list, label_dict


def split_instances_in_train_dev_test(instances, TRAIN_RATIO=0.6, DEV_RATIO=0.15):
    # Group the instances by original tweet
    original_tweets = dict()
    original_tweets_list = list()
    # candidate_chunk :: tokenized_tweet_with_masked_q_token :: question_label :: tweet
    for _, _, _, tweet in instances:
        if tweet not in original_tweets:
            original_tweets[tweet] = 1
            original_tweets_list.append(tweet)
        else:
            original_tweets[tweet] += 1

    train_size = int(len(original_tweets_list) * TRAIN_RATIO)
    dev_size = int(len(original_tweets_list) * DEV_RATIO)
    train_tweets = original_tweets_list[:train_size]
    dev_tweets = original_tweets_list[train_size:train_size + dev_size]
    test_tweets = original_tweets_list[train_size + dev_size:]
    segment_instances = {"train": list(), "dev": list(), "test": list()}
    # A dictionary that stores the segment each tweet belongs to
    tweets_to_segment = dict()
    for tweet in train_tweets:
        tweets_to_segment[tweet] = "train"
    for tweet in dev_tweets:
        tweets_to_segment[tweet] = "dev"
    for tweet in test_tweets:
        tweets_to_segment[tweet] = "test"
    # Get instances
    for instance in instances:
        tweet = instance[3]
        segment_instances[tweets_to_segment[tweet]].append(instance)

    return segment_instances['train'], segment_instances['dev'], segment_instances['test']


def get_statistic(data, classes_list):
    statistic_dict = {idx: [type, 0] for idx, type in enumerate(classes_list)}
    data_len = len(data)
    split_idx = int(data_len * 0.6) + int(data_len * 0.15)
    for instance in data[split_idx:]:
        statistic_dict[instance[2]][1] += 1
    return statistic_dict


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str,
                    required=True)
parser.add_argument("-t", "--task", help="Event for which we want to train the baseline", type=str, required=True)
parser.add_argument("-s", "--save_directory", help="Path to the directory where we will save model and the tokenizer",
                    type=str, required=True)
parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=32)
parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=8)
parser.add_argument("-l", "--lr", help="Learning rate", type=float, default=2e-5)
args = parser.parse_args()


def get_metric(gold_labels, predicted_probs, label_name_dict, threshold_list):
    label_len = len(label_name_dict)

    # f1::threshold::tn::fp::fn::tp
    metric_dict = {i: [0, 0] for i in range(label_len)}

    np_gold_labels = np.asarray(gold_labels)
    np_pre_probs = np.asarray(predicted_probs)
    for i in range(label_len):
        gold_label_i = np_gold_labels[:, i]
        pre_label_i = np_pre_probs[:, i]
        for threshold in threshold_list:
            f1_i = get_f1(gold_label_i, pre_label_i, threshold)
            # update best f1
            if f1_i > metric_dict[i][0]:
                metric_dict[i][0] = f1_i
                metric_dict[i][1] = threshold

    for i, f1_thre in metric_dict.items():
        print("slot: {}  best dev f1: {}  threshold: {}".format(label_name_dict[i], f1_thre[0], f1_thre[1]))

    return metric_dict


def get_f1(gold_label, pre_probs, threshold):
    pre_label = (pre_probs > threshold).astype(int)
    f1 = f1_score(gold_label, pre_label)
    return f1


def get_test_result(gold_labels, predicted_probs, metric_dict, label_name_dict):
    label_len = len(label_name_dict)
    np_gold_labels = np.asarray(gold_labels)
    np_pre_probs = np.asarray(predicted_probs)
    result_dict = {label: dict() for idx, label in label_name_dict.items()}

    for i in range(label_len):
        gold_label_i = np_gold_labels[:, i]
        pre_label_i = np_pre_probs[:, i]
        threshold = metric_dict[i][1]
        predicted_label_i = (pre_label_i > threshold).astype(int)
        f1 = f1_score(gold_label_i, predicted_label_i)
        tn, fp, fn, tp = confusion_matrix(gold_label_i, predicted_label_i).ravel()
        p = tp / (tp + fp)
        r = tp / (tp + fn)

        result_dict[label_name_dict[i]]['p'] = p.item()
        result_dict[label_name_dict[i]]['r'] = r.item()
        result_dict[label_name_dict[i]]['f1'] = f1.item()
        result_dict[label_name_dict[i]]['tp'] = tp.item()
        result_dict[label_name_dict[i]]['fp'] = fp.item()
        result_dict[label_name_dict[i]]['fn'] = fn.item()
        result_dict[label_name_dict[i]]['tn'] = tn.item()
        result_dict[label_name_dict[i]]['slot #'] = (tp + fn).item()
        result_dict[label_name_dict[i]]['threshold'] = threshold
        print('slot type: ', label_name_dict[i])
        print('f1: {}'.format(f1))
        print('precision: {}'.format(p))
        print('recall: {}'.format(r))
        print('tp: ', tp)
        print('fp: ', fp)
        print('fn: ', fn)
        print('golden # of slot:', tp + fn, '\n')

    return result_dict


def calc_micro_f1(tp, fp, fn):
    tp = sum(tp)
    fp = sum(fp)
    fn = sum(fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print(f1)
    return f1


def main():
    # Read all the data instances

    data, label_dict = generate_instances(args.data_file)
    label_name_dict = {num: name for name, num in label_dict.items()}

    # Create the save_directory if not exists
    make_dir_if_not_exists(args.save_directory)

    # Initialize tokenizer and model with pretrained weights
    tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
    config = BertConfig.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
    config.num_labels = len(label_dict)

    # print(config)
    model = MultiLabelBertForCovidEntityClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert',
                                                                       config=config)

    # Add new tokens in tokenizer
    new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>", "<URL>", "@USER"]}
    tokenizer.add_special_tokens(new_special_tokens_dict)

    # Add the new embeddings in the weights
    print("Embeddings type:", model.bert.embeddings.word_embeddings.weight.data.type())
    print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
    embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
    new_embeddings = torch.FloatTensor(len(new_special_tokens_dict["additional_special_tokens"]),
                                       embedding_size).uniform_(-0.1, 0.1)
    # new_embeddings = torch.FloatTensor(2, embedding_size).uniform_(-0.1, 0.1)
    print("new_embeddings shape:", new_embeddings.size())
    new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data, new_embeddings), 0)
    model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
    print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
    # Update model config vocab size
    model.config.vocab_size = model.config.vocab_size + len(new_special_tokens_dict["additional_special_tokens"])
    model.to(device)

    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

    model_config = dict()

    # Split the data into train, dev and test and shuffle the train segment
    train_data, dev_data, test_data = split_instances_in_train_dev_test(data)
    random.shuffle(train_data)  # shuffle happens in-place

    # Load the instances into pytorch dataset
    train_dataset = COVID19TaskDataset(train_data)
    dev_dataset = COVID19TaskDataset(dev_data)
    test_dataset = COVID19TaskDataset(test_data)

    tokenize_collator = TokenizeCollator(tokenizer, entity_start_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0,
                                  collate_fn=tokenize_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0,
                                collate_fn=tokenize_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0,
                                 collate_fn=tokenize_collator)

    optimizer = AdamW(model.parameters(), args.lr, eps=1e-8)

    # Number of training epochs.
    epochs = args.n_epochs

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy.
    training_stats = []

    # Find the accumulation steps
    accumulation_steps = args.batch_size / POSSIBLE_BATCH_SIZE

    # Loss trajectory for epochs
    epoch_train_loss = list()

    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)

        # Reset the total loss for each epoch.
        total_train_loss = 0
        train_loss_trajectory = list()

        # Reset timer for each epoch
        start_time = time.time()
        model.train()

        dev_log_frequency = 5
        n_steps = len(train_dataloader)
        dev_steps = int(n_steps / dev_log_frequency)
        for step, batch in enumerate(pbar):
            # Forward
            input_dict = {"input_ids": batch["input_ids"].to(device),
                          "entity_start_positions": batch["entity_start_positions"].to(device),
                          "labels": batch["gold_labels"].to(device),
                          "token_type_ids": batch["token_type_ids"].to(device),
                          "attention_mask": batch["attention_mask"].to(device)
                          }

            loss, logits = model(**input_dict)

            # Accumulate loss
            total_train_loss += loss.item()

            # Backward: compute gradients
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                # Calculate elapsed time in minutes and print loss on the tqdm bar
                elapsed = format_time(time.time() - start_time)
                avg_train_loss = total_train_loss / (step + 1)
                # keep track of changing avg_train_loss
                train_loss_trajectory.append(avg_train_loss)
                pbar.set_description(
                    f"Epoch:{epoch + 1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters
                optimizer.step()

                # Clean the model's previous gradients
                model.zero_grad()  # Reset gradients tensors

                # Update the learning rate.
                scheduler.step()
                pbar.update()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - start_time)

        # Record all statistics from this epoch.
        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time})

        # Save the loss trajectory
        epoch_train_loss.append(train_loss_trajectory)

    log_list(training_stats)

    # Save the model and the Tokenizer here:

    model.save_pretrained(args.save_directory)
    # Save each subtask classifiers weights to individual state dicts
    tokenizer.save_pretrained(args.save_directory)

    # Plot the train loss trajectory in a plot
    # train_loss_trajectory_plot_file = os.path.join("myTest", "train_loss_trajectory.png")

    # TODO: Plot the validation performance
    # Save dev_subtasks_validation_statistics

    # Save the model name in the model_config file
    model_config["model"] = "MultiLabel"
    model_config["epochs"] = args.n_epochs

    # Find best threshold for each subtask based on dev set performance
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # test_predicted_labels, test_gold_labels = make_predictions_on_dataset(test_dataloader,model, device, 'tested_positive', True)

    dev_predicted_labels, dev_gold_labels, dev_predicted_probs = make_predictions_on_dataset(dev_dataloader, model,
                                                                                             device, args.task + "dev",
                                                                                             True)

    dev_metric_dict = get_metric(dev_gold_labels, dev_predicted_probs, label_name_dict, thresholds)

    # Evaluate on Test

    predicted_labels, gold_labels, predicted_probs = make_predictions_on_dataset(test_dataloader, model, device,
                                                                                 'tested_positive')
    result_dict = get_test_result(gold_labels, predicted_probs, dev_metric_dict, label_name_dict)

    # calc micro f1
    tp_list = list()
    fp_list = list()
    fn_list = list()
    f1_list = list()
    for slot, data_dict in result_dict.items():
        if args.task == 'tested_negative' and slot in ['name', 'relation', 'where', 'gender_male', 'gender_female']:
            tp_list.append(data_dict['tp'])
            fp_list.append(data_dict['fp'])
            fn_list.append(data_dict['fn'])
            f1_list.append(data_dict['f1'])
        elif args.task == 'can_not_test' and slot in ['name', 'relation', 'where', 'symptoms']:
            tp_list.append(data_dict['tp'])
            fp_list.append(data_dict['fp'])
            fn_list.append(data_dict['fn'])
            f1_list.append(data_dict['f1'])
        elif args.task == 'death' and slot in ['name', 'relation', 'when', 'where', 'age']:
            tp_list.append(data_dict['tp'])
            fp_list.append(data_dict['fp'])
            fn_list.append(data_dict['fn'])
            f1_list.append(data_dict['f1'])
        elif args.task in ['tested_positive', 'cure_and_prevention']:
            tp_list.append(data_dict['tp'])
            fp_list.append(data_dict['fp'])
            fn_list.append(data_dict['fn'])
            f1_list.append(data_dict['f1'])

    micro_f1 = calc_micro_f1(tp_list, fp_list, fn_list)
    macro_f1 = np.mean(f1_list)

    print('The micro f1 score: ', micro_f1)
    print('The macro f1 score: ', macro_f1)
    result_dict['micro f1 score'] = micro_f1
    result_dict['macro f1 score'] = macro_f1
    result_dict['epoch'] = args.n_epochs
    result_dict['batch_size'] = args.batch_size

    # Save model_config and results
    model_config_file = os.path.join(args.save_directory, "model_config.json")
    results_file = os.path.join(args.save_directory,
                                args.task + '_e' + str(args.n_epochs) + '_b' + str(args.batch_size) + "_results.json")

    save_in_json(model_config, model_config_file)
    save_in_json(result_dict, results_file)


if __name__ == '__main__':
    main()
    print('done')
