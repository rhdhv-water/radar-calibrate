from setuptools import setup

description_long = '\n\n'.join([
    open('README.rst').read(),
    ])

setup(name='radar-calibrate',
      version='0.1',
      description=description_long,
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='GPL',
      packages=['radar-calibrate'],
      install_requires=[
          'PyKrige',
      ],
      zip_safe=False)