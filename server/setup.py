import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from setuptools import setup, find_packages

requires = [
    'Flask',
    'flask_cors',
    'tensorflow',
    'pandas',
    'keras',
    'scikit-image'
]

setup(
    name='howhotami',
    version='0.0',
    description='Marketing for Electric Brain',
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
    keywords='web pyramid pylons',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require={

    },
    package_data={
        "howhotami": [
            "data/ratings.csv",
            "data/images/*.jpg"
        ],
    },
    install_requires=requires,
    dependency_links=[
    ],
    entry_points={
        'paste.app_factory': [
            'main = fathion:main',
        ],
        'console_scripts': [
            'howhotami_train = howhotami.train:trainModel',
        ]
    },
)
