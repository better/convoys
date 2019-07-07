from setuptools import setup, find_packages

long_description = '''
For more information, see
`the package documentation <https://better.engineering/convoys>`_
or
`the Github project page <https://github.com/better/convoys>`_.

.. image:: https://better.engineering/convoys/_images/dob-violations-combined.png
'''

setup(name='convoys',
      version='0.1.3',
      description='Fit machine learning models to predict conversion using Weibull and Gamma distributions',
      long_description=long_description,
      url='https://better.engineering/convoys',
      license='MIT',
      author='Erik Bernhardsson',
      author_email='erikbern@better.com',
      packages=find_packages(),
      install_requires=[
          'autograd',
          'autograd-gamma',
          'emcee',
          'matplotlib>=2.0.0',
          'pandas',
          'numpy',
          'scipy',
      ])
