# Copyright (C) 2025-present Naver Corporation. All rights reserved.
from setuptools import setup, find_packages

curope_dep = ['curope @ git+https://github.com/naver/croco.git@croco_module#egg=curope&subdirectory=curope']
optional_dep = [
    'pillow-heif'
]

setup(
    name="must3r",
    version="1.0.0",
    packages=find_packages(include=["must3r", "must3r.*"]),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'numpy',
        'numpy-quaternion',
        'opencv-python',
        'einops',
        'tensorboard',
        'h5py',
        'pillow',
        'roma',
        'gradio>=5.0.0',
        'scipy',
        'trimesh',
        'pyglet<2',
        'huggingface-hub[torch]>=0.22',
        'cython',
        'pyaml',
        'open3d',
        'viser>=1.0.0',
        'croco @ git+https://github.com/naver/croco.git@croco_module#egg=croco',
        'dust3r @ git+https://github.com/naver/dust3r.git@dust3r_setup#egg=dust3r',
        "asmk[cpu] @ git+https://github.com/lojzezust/asmk.git"
    ],
    python_requires=">=3.11",
    extras_require={
        "curope": curope_dep,
        "optional": optional_dep,
        "all": curope_dep + optional_dep

    },
    entry_points={
        'console_scripts': [
            'must3r_demo=must3r.demo.gradio:main',
            'must3r_slam=must3r.slam.slam:main'
        ]
    }
)
