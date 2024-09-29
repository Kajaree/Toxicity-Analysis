from setuptools import find_packages, setup

setup(
    name='unintended_bias_mitigation',
    version='0.0.1',
    author="",
    description="",
    long_description="",
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["torch",
                      "torchvision",
                      "flair",
                      "h5py",
                      "torchtext",
                      "tqdm",
                      "nltk >=3.4.5",
                      "flair",
                      "datastack",
                      "mlgym",
                      "outlier-hub"
                      ],
    python_requires=">=3.7"
)
