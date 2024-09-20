from setuptools import setup, find_packages
import versioneer
PACKAGES = find_packages()

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

opts = dict(name='clustcrdist',
            maintainer='Sebastiaan Valkiers',
            maintainer_email='sebastiaan.valkiers@uantwerpen.be',
            description='Pairwise distance calculation and neighborhood enrichment analysis of TCR repertoires',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/svalkiers/immune_response_detection',
            license='MIT',
            author='Sebastiaan Valkiers',
            author_email='sebastiaan.valkiers@uantwerpen.be',
            version=versioneer.get_version(),
            cmdclass=versioneer.get_cmdclass(),
            packages=PACKAGES,
            python_requires='>=3.9',
            classifiers=[
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.9',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
            ],
            include_package_data=True,  # This ensures non-code files are included
            package_data={
                'clustcrdist': [
                    'constants/data/*.tsv', 
                    'constants/data/*.csv', 
                    'constants/data/*.txt',
                    'constants/modules/*.py',],
            },
            entry_points={
                'console_scripts': [
                    'clustcrdist=clustcrdist.run_pipeline:main',  # This defines the 'clustcrdist' command
        ],
        },
        )

install_reqs = [
      'numpy>=1.23.5',
      'pandas>=1.5.2',
      'faiss-cpu==1.7.3',
      'scipy==1.9.3', 
      'scikit-learn==1.2.0',
      'pynndescent==0.5.8',
      'igraph==0.10.3',
      'networkx==3.0',
      'olga==1.2.4',
      'leidenalg==0.9.1',
      'logomaker==0.8',
      'statsmodels==0.14.0',
      'matplotlib>=3.6.2',
      'seaborn>=0.13.2',
      'biopython>=1.84',
      'parmap==1.6.0',
      'pip'
      ]

if __name__ == "__main__":
      setup(**opts, install_requires=install_reqs)
