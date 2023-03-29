import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoML",
    version="2.1",
    author="Owen ODriscoll",
    author_email="owenodriscoll1@googlemail.com",
    description="Python package for automated hyperparameter-optimization of common machine-learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/owenodriscoll/AutoML",
    packages=setuptools.find_packages(),
    python_requires='>=3.9.12',
)
