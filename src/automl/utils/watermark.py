import tomli
from IPython import get_ipython
from packaging.requirements import Requirement

def watermark_from_pyproject(pyproject_path="../pyproject.toml"):
    """
    Load dependencies from a pyproject.toml file and display them using the watermark extension.

    This function extracts the dependencies listed in the pyproject.toml file, removes any extras (e.g., `[extra]`),
    and uses the `watermark` IPython extension to display the dependencies in the current environment.

    Parameters
    ----------
    pyproject_path : str, optional
        The path to the pyproject.toml file (default is "../pyproject.toml").

    Returns
    -------
    None
        Prints the dependencies using the watermark extension or a message if not in an IPython environment.

    Examples
    --------
    >>> watermark_from_pyproject("../pyproject.toml")
    """
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    packages = pyproject['project']['dependencies']
    cleaned_packages = [Requirement(p).name for p in packages]
    package_str = ",".join(cleaned_packages)

    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic("load_ext", "watermark")
        command = f"-n -u -v -iv -w -p {package_str}"
        ipython.run_line_magic("watermark", command)
    else:
        print("This function must be run in an IPython or Jupyter environment.")
