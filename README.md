# Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention (DBSA)

This repository contains the code for reproducing the experiments from the preprint "Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention" It is built on code from the paper [In-Context Learning with Long-Context Models: An In-Depth Exploration.](https://arxiv.org/abs/2405.00200).  


## Enviornment setup
- create conda env with python 3.9
- install requirements according to env.yml
- to run DBSA, replace `modeling_llama.py` with `replacement_modeling_llama.py`, and `modeling_utils.py` with `replacement_modeling_utils.py`. You can find the path with:
  ```bash
  python -c "import transformers; print(transformers.__path__)"

## Experiments

To run the experiments from the paper, use the `run_evaluation` script with appropriate arguments. 

- to run main experiments, run `exp-main-*.sh`
- to run ablation experiments, run `exp-abl-*.sh`
- to run finetuning experiments, we provide the training script in finetuning folder

## Citation


## Acknowledgments
This codebase builds upon the following papers
```
@misc{bertsch2024incontext,
      title={In-Context Learning with Long-Context Models: An In-Depth Exploration}, 
      author={Amanda Bertsch and Maor Ivgi and Uri Alon and Jonathan Berant and Matthew R. Gormley and Graham Neubig},
      year={2024},
      eprint={2405.00200},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{ratner2023parallel,
      title={Parallel Context Windows for Large Language Models}, 
      author={Nir Ratner and Yoav Levine and Yonatan Belinkov and Ori Ram and Inbal Magar and Omri Abend and Ehud Karpas and Amnon Shashua and Kevin Leyton-Brown and Yoav Shoham},
      year={2023},
      eprint={2212.10947},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
