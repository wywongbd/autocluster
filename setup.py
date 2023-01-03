from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name = 'autocluster',
    version = '0.5.2',
    description = 'Automated machine learning toolkit for performing clustering tasks.',
    long_description_content_type = "text/markdown",
    long_description = README,
    license = 'BSD-3-clause',
    packages = find_packages(),
    author = 'Wong Wen Yan',
    author_email = 'wywongbd@gmail.com',
    keywords = ['automl', 'clustering', 'bayesian-optimization', 'hyperparameter-optimization'],
    url = 'https://github.com/wywongbd/autocluster',
    download_url = 'https://pypi.org/project/autocluster/'
)

install_requires = [            
    'joblib',
    'cycler',
    'kiwisolver',
    'matplotlib',
    'numpy',
    'pandas',
    'pyparsing',
    'python-dateutil',
    'pytz',
    'scikit-learn',
    'scipy',
    'six',
    'sklearn'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
