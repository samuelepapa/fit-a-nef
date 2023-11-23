import setuptools

setuptools.setup(
    name="fit-a-nef",
    version="0.1",
    author="Samuele Papa",
    author_email="samuele.papa@gmail.com",
    description="A package to load and transform neural datasets",
    url="https://github.com/samuelepapa/neural-field-arena",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax[cuda11_pip]>=0.4.13",
        "jaxlib==0.4.20",
        "optax>=0.1.5",
        "flax>=0.7.0",
        "torchvision>=0.15",
        "torchaudio>=2.0",
        "torch@https://download.pytorch.org/whl/cpu",
        "h5py>=3.0.0",
        "numpy>=1.19.5",
        "absl-py>=0.12.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.16"],
    },
)
