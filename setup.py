import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pwlfit",
    author="David Kirkby",
    author_email="dkirkby@uci.edu",
    description="Package for piecewise linear fitting of noisy data",
    keywords="linear fitting, piecewise linear, astronomy, data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dkirkby/pwlfit",
    project_urls={
        "Documentation": "https://github.com/dkirkby/pwlfit",
        "Bug Reports": "https://github.com/dkirkby/pwlfit/issues",
        "Source Code": "https://github.com/dkirkby/pwlfit",
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "pwlfit.data": [
            "sampleA.json",
            "sampleB.json",
            "sampleC.json",
        ],
    },
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["numpy","scipy","pyyaml"],
    extras_require={
        "dev": ["check-manifest"],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=bed:main',
    # You can execute `run` in bash to run `main()` in src/bed/__init__.py
    #     ],
    # },
)
