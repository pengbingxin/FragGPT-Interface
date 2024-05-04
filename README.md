# FragGPT

FragGPT：Unlocking comprehensive molecular design across all scenarios with large language model and unordered chemical language

## data



## Environment

- python = 3.8.3
- pytroch =1.13.1 + pt113cu116
- RDKit
- transformers=4.33.0
- peft=0.5.0

```
pip install requirements.txt  

sudo apt-get install python3-dev

cd ./admet_predictor/datasets

python setup.py build_ext --inplace

## python generate_frag_pocket.py
```


## run
```
python generate_frag_pocket.py --n_generated  400000 \
                                --generate_mode  rgroup \
                                --pdb  ./data/MolecularFactory/8gng/8GNG-protein.pdb \
                                --sdf  ./data/MolecularFactory/8gng/8GNG-ligand.sdf \
                                --reference 'O=C1[C@H](N=C(C2=CC=CC=C2N1)C3=CC=CC=C3)N[*]'  \
                                --save_dir .FragGPT_8gng2.csv \
                                --model_ckpt pretrain.pt \
                                --config  smiles_frag  \
                                --lora


```

