from setuptools import setup

setup(
    name='housing',
    version='0.1.0',    
    description='A library for predicting the housing price',
    author='Saravanan',
    author_email='saravanan.chinnu@tigeranalytics.com',
    license='MIT license',
    packages=['housing'],
    install_requires=['pandas',
                      'numpy','scipy','sklearn','tarfile'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
