# LiteTCI · Paper data

LiteTCI learns to rank scientific papers by whether experts recorded a *Technical Advance* in the paper. During training, a lightweight encoder learns from expert labels and large language model scores. At inference, it scores papers from their titles and abstracts. This repository contains the data behind the main-text tables and inference figure.

## Find a result

| Manuscript item | Files |
| --- | --- |
| Table 1 · Data splits | [Cohort counts](results/table1_cohorts.csv) and [paper identifiers, years and splits](data/paper_splits.csv) |
| Table 2 · Model comparison | [Parameter counts and average precision](results/table2_performance.csv) |
| Table 3 · Teacher supervision | [Label-only, permuted-score and paper-specific supervision](results/table3_supervision.csv) |
| Table 4 · Abstract wording | [Subgroup results](results/table4_abstract_subgroups.csv) and [exact matching rule](data/abstract_rule.json) |
| Figure 2 · Inference efficiency | [Latency, throughput and peak allocated GPU memory for each batch size](data/figure2_efficiency.csv) |

The [per-paper scores](data/evaluation_scores.csv) cover the 548 validation papers and 687 test papers in Tables 2–4. Higher scores rank first within each model and split. `student_labels_only` identifies the matched training control in Table 3. `scibert` identifies the separately fine-tuned baseline in Table 2. The *Cell* records use the original Scopus IDs and include PMIDs for lookup. For the 839 other-journal papers described in Table 1, records are keyed by PMID.

Figure 2 uses the same 512 validation papers at every batch size. `figure2_order` records their processing order. Timing includes tokenization through score return with both models already loaded on an Intel Arc B580 (12 GB). The separate capacity check in the text used 320 tokens per input and found a largest successful LiteTCI batch of 759 papers.

## Recompute the tables

Download the original *Cell* expert labels from [Jeon et al., `papers_all_tags.csv`](https://github.com/nero96in/paper_novelty/blob/857a53fa4db96805f10d81e4d9e63c6d6ef1cba9/data/papers_all_tags.csv). A positive paper has at least one *Technical Advance* assignment. All included papers received at least one expert classification.

```bash
python -m pip install numpy scipy scikit-learn
python reproduce.py --cell-tags papers_all_tags.csv
```

Add `--bootstrap` to recompute the paired interval in Section 5.3. To check the cross-journal count in Table 1, add `--cross-tags tags-1990-2000.csv` using the [original CSV](https://dataverse.harvard.edu/api/access/datafile/6858656?format=original) from [Shi and Evans, Harvard Dataverse](https://doi.org/10.7910/DVN/NFSYYA). To rebuild the Table 4 groups from the abstracts, add `--cell-papers cell_all.csv` using [Jeon et al.’s article file](https://github.com/nero96in/paper_novelty/blob/857a53fa4db96805f10d81e4d9e63c6d6ef1cba9/data/papers/cell_all.csv).
