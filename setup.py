from setuptools import find_packages, setup
requirements = """
adnmtf==0.1.164
scipy==1.9.1 
"""

description = "Integrated Longitudinal Multi Source Model"

setup(
    name="adilsm",
    version='0.0.8',
    author="Mazars",
    author_email="victor.chemla@mazars.fr",
    description=f"{description}",
    packages=find_packages(),
    install_requires=requirements,
)
