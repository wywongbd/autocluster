from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name = 'autocluster',
    version = '0.3.1',
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
    'cycler>=0.10.0',
    'joblib>=0.11.0',
    'kiwisolver>=1.0.1',
    'matplotlib==3.0.3',
    'numpy>=1.12.0',
    'pandas==0.24.2',
    'pyparsing==2.4.0',
    'python-dateutil>=2.5.0',
    'pytz==2019.1',
    'scikit-learn==0.21.3',
    'scipy>=0.17.0',
    'six>=1.5.0',
    'sklearn==0.0'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
