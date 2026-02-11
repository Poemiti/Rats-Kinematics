from setuptools import setup, find_packages


setup(
    name="rats_kinematics_utils",
    version="1.1",
    license="MIT",
    author="Poemiti Duprat",
    author_email="poemiti.duprat@gmail.com",
    description="Utility modules to make analysis of Rats kinematics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)



# how to make it a package : 
# run the following commande : python -m pip install -e .
