from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='enviroclass',
      version="0.0.12",
      description="EnviroClass Model (api_pred)",
      license="MIT",
      author="F. Haisch, T. Hempel, L. Berger, B. Dabholkar",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
