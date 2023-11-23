To install the dependencies, the following worked on both a machine with CUDA 11.8 and a machine with CUDA 12.0:

```bash
pip3 install torch torchvision torchaudio torchdata==0.7.0 --index-url https://download.pytorch.org/whl/cpu

pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install jraph flax

pip install faiss-gpu scikit-learn

pip install wandb optuna h5py trimesh ml_collections cython

cd dataset/shape_dataset/utils

python setup.py build_ext --inplace
```

For creating the plots:

```
pip install matplotlib seaborn pandas
```
