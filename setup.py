from distutils.core import setup

setup(
    # How you named your package folder (MyLib)
    name = 'autocluster',
    
    # Chose the same as "name"
    packages = ['autocluster'],   
    
    # Start with a small number and increase it with every change you make
    version = '0.1',      
    
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license = 'MIT',    
    
    # Give a short description about your library
    description = 'Automated machine learning toolkit for performing clustering tasks.', 
    
    # Type in your name
    author = 'Wong Wen Yan',        
    
    # Type in your E-Mail
    author_email = 'wywongbd@gmail.com',      
    
    # Provide either the link to your github or to your website
    url = 'https://github.com/wywongbd/autocluster',   
    
    # TODO: include url to download the code
    download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    
    
    # Keywords that define your package best
    keywords = ['automl', 'clustering', 'bayesian-optimization', 'hyperparameter-optimization'],   
    
    # dependencies
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
    ],
    
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',    
        
        # Define that your audience are developers
        'Intended Audience :: Developers',      
        
        # Again, pick a license
        'License :: OSI Approved :: MIT License',   
        
        #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)