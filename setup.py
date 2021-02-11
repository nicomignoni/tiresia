import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiresia",
    version="0.0.11",
    author="Nicola Mignoni",
    author_email="nicola.mignoni@gmail.com",
    description="Tiny sklearn-based AutoML tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicomignoni/tiresia",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["tiresia"],
    python_requires='>=3.6'
)
