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
        'sklearn',
        'scikit-learn',
        'numpy',
        'pandas'
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