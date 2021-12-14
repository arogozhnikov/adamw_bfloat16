from setuptools import setup

setup(
    name="adamw_bfloat16",
    version='0.1.0',
    description="Stable bfloat16-only optimizer in pytorch, compatible with ",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arogozhnikov/adamw_bfloat16',
    author='Alex Rogozhnikov',

    packages=['adamw_bfloat16'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],
    keywords='optimization, pytorch',
    install_requires=[
        'torch'
    ],
)