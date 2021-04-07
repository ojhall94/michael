import setuptools
import os

version = {}

with open(os.path.join(*['michael','version.py'])) as fp:
	exec(fp.read(), version)

setuptools.setup(
    name="Michael",
    version=version['__version__'],
    author="Oliver Hall",
    author_email="oliver.hall@esa.int",
    description="A package for extracting rotation rates from TESS FFIs",
    long_description=open("README.md").read(),
    packages=['michael'],
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    license="MIT",
)
