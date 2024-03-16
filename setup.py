import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="latent_experiments",  
    version="0.0.8",  
    author="Amirhossein Nakhaei",  
    description="A package for running and plotting latent experiments.",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),  # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  
    python_requires=">=3.6",  
    py_modules=["latent_experiments/src"],  
    package_dir={"": "src"}, 
    install_requires=[],  
)
