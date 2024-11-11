from setuptools import setup, find_packages

setup(
    name='simplified_pointnet_for_segmentation',
    version='1.0',
    description='A project using PointNet for segmentation tasks on ShapeNetCore dataset',
    author='Tripmasterwang',
    author_email='wangys24@mails.tsinghua.edu.cn',
    packages=find_packages(),
    install_requires=[
        'tqdm',                    # Progress bar
        'numpy',                   # Numerical operations
        'argparse'                 # For command-line argument parsing
    ],
    python_requires='>=3.6',
)