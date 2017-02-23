from setuptools import setup, find_packages

version = '0.1.4'

setup(
    name='padua',
    version=version,
    url='http://github.com/mfitzp/padua',
    author='Martin Fitzpatrick',
    author_email='martin.fitzpatrick@gmail.com',
    description='A Python interface for Proteomic Data Analysis, working with MaxQuant & Perseus outputs',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Desktop Environment',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Widget Sets',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'statsmodels',
        'matplotlib-venn',
        'scikit-learn',
        'requests',
        'requests_toolbelt',
    ]
)
