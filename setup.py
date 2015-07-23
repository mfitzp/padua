from setuptools import setup, find_packages

version = '0.0.2'

setup(
    name='pyMaxQuant',
    version=version,
    url='http://github.com/mfitzp/pymaxquant',
    author='Martin Fitzpatrick',
    author_email='martin.fitzpatrick@gmail.com',
    description='A Python interface for working with MaxQuant & Perseus outputs',
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
)
