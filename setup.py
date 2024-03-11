from setuptools import find_packages, setup
import versioneer
requirements = """
setuptools>=61
adnmtf==0.1.164
"""

setup(
    name="adilsm",
    version='0.0.1',
    author="Mazars",
    author_email="victor.chemla@mazars.fr",
    description="{description}",
    packages=find_packages(),
    install_requires=requirements,
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)
