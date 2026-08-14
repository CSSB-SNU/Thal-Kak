# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by CSSB Thal-Kak (2026):
#   - Replaced the fair-esm backend with HuggingFace transformers for ESM2
#     embeddings (HF AutoModel/AutoTokenizer; drop-in FastaBatchedDataset and
#     batch converter; hidden-states extraction).
#     Reason: in the unified env, fair-esm and the EvolutionaryScale `esm`
#     package (required by ESMFold2) both claim the top-level `esm` namespace
#     and cannot coexist. ESMFold2 needs the EvolutionaryScale `esm`, so the
#     fair-esm import here was removed to resolve the clash.
# The above modifications are likewise licensed under the Apache License 2.0.

import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer  # CSSB Thal-Kak (2026): HF transformers backend (replaces `from esm import ...`); avoids the `esm` namespace clash with ESMFold2's EvolutionaryScale esm

# CSSB Thal-Kak (2026): config keyed by HuggingFace `hf_id` instead of fair-esm `model_path`.
ESM_CONFIG = {
    "esm2-3b": {
        "type": "esm2",
        "hf_id": "facebook/esm2_t36_3B_UR50D",
        "emb_dim": 2560,
        "n_layers": 36,
    },
    "esm2-3b-ism": {
        "type": "esm2",
        "hf_id": None,
        "emb_dim": 2560,
        "n_layers": 36,
    },
}


class FastaBatchedDataset(torch.utils.data.Dataset):
    """Drop-in replacement for fair-esm's FastaBatchedDataset: indexed access
    yields (label, sequence) tuples and `get_batch_indices` reproduces the
    same length-sorted greedy packing."""

    def __init__(self, labels, sequences):
        self.labels = list(labels)
        self.sequences = list(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.labels[idx], self.sequences[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = sorted(
            (len(s) + extra_toks_per_seq, i) for i, s in enumerate(self.sequences)
        )
        batches, cur, cur_sz = [], [], 0
        for sz, idx in sizes:
            new_sz = max(cur_sz, sz)
            if cur and (len(cur) + 1) * new_sz > toks_per_batch:
                batches.append(cur)
                cur, cur_sz = [], 0
            cur.append(idx)
            cur_sz = max(cur_sz, sz)
        if cur:
            batches.append(cur)
        return batches


def _make_batch_converter(tokenizer, truncation_seq_length):
    """Closure mirroring fair-esm's BatchConverter (labels, strs, toks)
    signature. HF EsmTokenizer prepends CLS (id 0), appends EOS (id 2),
    pads with id 1 -- same indices as fair-esm's alphabet, so `toks` is
    bit-identical to the fair-esm output."""

    def convert(raw_batch):
        labels, strs = zip(*raw_batch)
        encoded = tokenizer(
            list(strs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=truncation_seq_length + 2,
        )
        return list(labels), list(strs), encoded["input_ids"]

    return convert


def _load_esm2_model(hf_id):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    return model, tokenizer


def load_esm_model(model_name, local_esm_dir="release_data/checkpoint"):
    hf_id = ESM_CONFIG[model_name].get("hf_id")
    if hf_id is None:
        raise RuntimeError(
            f"{model_name} weights are only distributed as a fair-esm format .pt "
            "(manual download from https://af3-dev.tos-cn-beijing.volces.com/release_model/). "
            "This loader uses HuggingFace transformers and cannot read that format. "
            "Choose esm2-3b instead."
        )
    if model_name.startswith("esm2"):
        model, alphabet = _load_esm2_model(hf_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, alphabet


def _check_files_exist(save_dir, labels):
    return all(
        [os.path.exists(os.path.join(save_dir, label + ".pt")) for label in labels]
    )


def compute_ESM_embeddings(
    model_name,
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
):
    if model_name.startswith("esm2"):
        embeddings = compute_esm2_embeddings(
            model,
            alphabet,
            labels,
            sequences,
            save_dir,
            toks_per_batch,
            truncation_seq_length,
        )
    return embeddings


# Adapt from Corso, Gabriele, et al. "Diffdock: Diffusion steps, twists, and turns for molecular docking."
# URL: https://github.com/gcorso/DiffDock/blob/main/utils/inference_utils.py
def compute_esm2_embeddings(
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
):
    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=_make_batch_converter(alphabet, truncation_seq_length),
        batch_sampler=batches,
    )
    repr_layer = model.config.num_hidden_layers
    embeddings = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if _check_files_exist(save_dir, labels):
                continue
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(input_ids=toks, output_hidden_states=True)
            representation = out.hidden_states[repr_layer].to(device="cpu")
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representation[i, 1 : truncate_len + 1].clone()
                save_path = os.path.join(save_dir, label + ".pt")
                torch.save(embeddings[label], save_path)
    return embeddings


def pdb_sequences_iterator(
    input_path="./scripts/msa/data/pdb_seqs/pdb_seq.csv",
    save_path="./scripts/msa/data/pdb_seqs/pdb_labels_seqs.csv",
    start_id=0,
    end_id=-1,
):
    if os.path.exists(save_path):
        df_seq = pd.read_csv(save_path)
    else:
        df = pd.read_csv(input_path)
        # Protein only
        df = df[df["mol_type"] == "protein"]
        # Sequence name
        df["pdb_entity_id"] = df["pdb_id"] + "_" + df["entity_id"].astype(str)
        # Group by 'seq'
        df_seq = (
            df.groupby("seq")["pdb_entity_id"]
            .apply(lambda x: ",".join(x))
            .reset_index()
        )
        # Use the first pdb_entity_id as the label
        df_seq["seq_label"] = df_seq["pdb_entity_id"].apply(lambda x: x.split(",")[0])
        assert df_seq["seq_label"].nunique() == len(df_seq)
        # Get a part id
        df_seq["part_id"] = df_seq["pdb_entity_id"].apply(lambda x: x[1:3])
        df_seq.to_csv(save_path)

    if end_id == -1:
        end_id = len(df_seq)
    df_seq = df_seq[start_id:end_id]

    part_counts = dict(df_seq["part_id"].value_counts())
    for part_id, count in part_counts.items():
        df_part = df_seq[df_seq["part_id"] == part_id]
        print(f"Part {part_id}: {len(df_part)} sequences.")
        yield part_id, df_part["seq_label"].tolist(), df_part["seq"].tolist()


def process_pdb_dataset(
    model_name, root_save_dir, pdb_seq_path, pdb_seq_label_path, start_id=0, end_id=-1
):

    model, alphabet = load_esm_model(model_name)
    seq_iterator = pdb_sequences_iterator(
        pdb_seq_path, pdb_seq_label_path, start_id, end_id
    )
    error_parts = []
    for part_id, labels, sequences in seq_iterator:
        save_dir = os.path.join(root_save_dir, f"{part_id}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[{part_id}] Generating ESM language model embeddings")
        lm_embeddings = compute_ESM_embeddings(
            model_name,
            model,
            alphabet,
            labels,
            sequences,
            save_dir,
            truncation_seq_length=4094,
            toks_per_batch=16384,
        )
        print(f"[{part_id}] Processed {len(lm_embeddings)} sequences in total. Done!")

    print("Error parts: ", error_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=list(ESM_CONFIG.keys()))
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    args = parser.parse_args()

    save_dir = f"./esm_embeddings/{args.model_name}"
    pdb_seq_path = "./scripts/msa/data/pdb_seqs/pdb_seq.csv"
    pdb_seq_label_path = "./scripts/msa/data/pdb_seqs/pdb_labels_seqs.csv"

    if not os.path.exists(save_dir):
        print("Make dir: ", save_dir)
        os.makedirs(save_dir)
    process_pdb_dataset(
        args.model_name,
        save_dir,
        pdb_seq_path,
        pdb_seq_label_path,
        args.start_id,
        args.end_id,
    )


if __name__ == "__main__":
    main()
