from setuptools import find_packages, setup

requirements = """
adnmtf==0.1.164
"""

setup(
    name="adilsm",
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "write_to": "version.txt",
        "root": ".",
        "relative_to": __file__,
    },
    author="Mazars",
    author_email="victor.chemla@mazars.fr",
    description="{description}",
    packages=find_packages(),
    install_requires=requirements,
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)
