import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


setuptools.setup(
    name='schema-to-json',
    author='Fan Bai',
    author_email='fan.bai@cc.gatech.edu',
    description='schema-to-json',
    keywords='information extraction, large-language-models, llm, prompting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/',
    project_urls={
        'Homepage': 'https://github.com/',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    install_requires=[
        'setuptools',
    ],
    include_package_data=True,
)
