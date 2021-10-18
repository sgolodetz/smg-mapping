from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-mapping",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="3D mapping systems",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-mapping",
    packages=find_packages(include=["smg.mapping", "smg.mapping.*"]),
    include_package_data=True,
    install_requires=[
        "smg-joysticks",
        "smg-open3d",
        "smg-pyoctomap",
        "smg-skeletons"
    ],
    extras_require={
        "all": [
            "smg-detectron2",
            "smg-pyremode"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
