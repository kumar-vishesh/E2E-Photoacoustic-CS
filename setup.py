from setuptools import setup, find_packages

setup(
    name='basicsr',
    version='0.1',
    description='Modified basicsr for End-to-End Photoacoustic Compressed Sensing',
    author='VK',
    packages=find_packages(
        exclude=('config', 'experiments', 'logs', 'tests', 'results')),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'pyyaml',
        'tqdm',
        'scipy',
    ],
    python_requires='>=3.8',
)
