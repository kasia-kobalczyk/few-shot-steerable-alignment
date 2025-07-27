import torch
from data.dataloaders import * 

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def unpack_pairs(pairs):
    input_ids = [d['pairs_input_ids'] for d in pairs]
    input_ids = torch.stack(input_ids)
    attention_mask = [d['pairs_attention_mask'] for d in pairs]
    attention_mask = torch.stack(attention_mask)
    cls_idxs = [d['pairs_cls_idxs'] for d in pairs]
    cls_idxs = torch.stack(cls_idxs)
    response_start_idxs = [d['pairs_response_start_idxs'] for d in pairs]
    response_start_idxs = torch.stack(response_start_idxs)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'cls_idx': cls_idxs,
        'response_start_idx': response_start_idxs
    }

def collate_fn(batch, cfg):
    context_data, target_data = zip(*batch)

    if cfg.target_datatype in ['embeddings', 'raw']:
        pairs_T = [d['pairs'] for d in target_data]
        pairs_T = torch.stack(pairs_T)
    elif cfg.target_datatype == 'tokens':
        pairs_T = unpack_pairs(target_data)

    if cfg.context_datatype in ['embeddings', 'raw']:
        pairs_C = [d['pairs'] for d in context_data]
        pairs_C = torch.stack(pairs_C)
    elif cfg.context_datatype == 'tokens':
        pairs_C = unpack_pairs(context_data)                                                                             

    choices_T = [d['choices'] for d in target_data]
    choices_T = torch.stack(choices_T)
    
    choices_C = [d['choices'] for d in context_data]
    choices_C = torch.stack(choices_C)

    labels_T = [d['label'] for d in target_data]
    labels_C = [d['label'] for d in context_data]
   
    return {
        'pairs_C': pairs_C,
        'choices_C': choices_C,
        'labels_C': labels_C,
        'pairs_T': pairs_T,
        'choices_T': choices_T,
        'labels_T': labels_T
    }
    

def get_dataloader(dataset, cfg):
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: collate_fn(batch, cfg), 
        drop_last=True,
        num_workers=1
    )
    return data_loader

def setup_dataloaders(cfg, splits=['train', 'val', 'test']):
    dataloaders = {}

    for split in splits:
        context_dataset = PreferenceDataset(
            path_to_data=f'{cfg.path_to_context_data}',
            split_file=f'{cfg.split_file}',
            split=split,
            datatype=cfg.context_datatype,
            labels=cfg.labels
        )
        target_dataset = PreferenceDataset(
            path_to_data=f'{cfg.path_to_target_data}',
            split_file=f'{cfg.split_file}',
            split=split,
            datatype=cfg.target_datatype,
            labels=cfg.labels
        )
        combined_dataset = ContextTargetDataset(cfg, split, context_dataset, target_dataset)
        dataloaders[split] = get_dataloader(combined_dataset, cfg)

    return dataloaders


def collect_pairs_choices(batch, num_context, min_num_context, max_num_context, num_targets, context_datatype):
    pairs_C, choices_C = batch['pairs_C'], batch['choices_C']
    pairs_T, choices_T = batch['pairs_T'], batch['choices_T']

    bs = pairs_C.shape[0]
    if num_context is None:
        num_context = np.random.randint(min_num_context, max_num_context + 1, 1)[0]

    context_idx = torch.tensor(
        np.random.choice(num_targets, num_context, replace=False)
    )

    if context_datatype == 'tokens':
        pairs_C_input_ids = torch.gather(
            pairs_C['input_ids'], 1, context_idx.view(1, -1, 1, 1).expand(bs, -1, 2, pairs_C['input_ids'].shape[-1])
        )
        pairs_C_attention_mask = torch.gather(
            pairs_C['attention_mask'], 1, context_idx.view(1, -1, 1, 1).expand(bs, -1, 2, pairs_C['attention_mask'].shape[-1])
        )
        pairs_C_cls_idxs = torch.gather(
            pairs_C['cls_idx'], 1, context_idx.view(1, -1, 1).expand(bs, -1, 2)
        )
        pair_C_response_start_idxs = torch.gather(
            pairs_C['response_start_idx'], 1, context_idx.view(1, -1, 1).expand(bs, -1, 2)
        )
        pairs_C = {
            'input_ids': pairs_C_input_ids,
            'attention_mask': pairs_C_attention_mask,
            'cls_idx': pairs_C_cls_idxs,
            'response_start_idx': pair_C_response_start
        }
    else:
        pairs_C = torch.gather(pairs_C, 1, context_idx.view(1, -1, 1, 1).expand(bs, -1, 2, pairs_C.shape[-1]))

    choices_C = torch.gather(choices_C, 1, context_idx.view(1, -1, 1).expand(bs, -1, 1))

    return pairs_C, choices_C, pairs_T, choices_T

if __name__ == '__main__':
    from argparse import Namespace
    cfg = Namespace(
        batch_size=16,
        min_num_context=0,
        max_num_context=10,
        num_targets=10,

        context_datatype='embeddings',
        target_datatype='tokens',
        path_to_context_data='data/ultra_feedback/embedded_pairs/meta-llama/Meta-Llama-3-8B',
        path_to_target_data='data/ultra_feedback/tokenized_pairs_512/meta-llama/Meta-Llama-3-8B',
        split_file='data/ultra_feedback/hh_pairs_conflict_1.0.csv',
        labels=['helpfulness', 'honesty']

        # context_datatype='raw',
        # target_datatype='raw',
        # path_to_data='data/synthetic_data',
        # split_file='synthetic.csv',
        # labels=['0', '1'],
       
    )

    train_dataloader = setup_dataloaders(cfg, splits=['train'])['train']
    for batch in train_dataloader:
        pairs_C, choices_C, pairs_T, choices_T = batch['pairs_C'], batch['choices_C'], batch['pairs_T'], batch['choices_T']
        print(pairs_C.shape, choices_C.shape)
        print(pairs_T['input_ids'].shape, pairs_T['attention_mask'].shape, pairs_T['cls_idx'].shape, choices_T.shape)
        assert batch['labels_C'] == batch['labels_T']
        break

