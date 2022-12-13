from setuptools import setup

setup(name='qspems',
      version='0.1',
      description='Network inference via process motifs',
      url='https://github.com/acuschwarze/network-inference-via-process-motifs',
      author='Alice Schwarze',
      author_email='alice.c.schwarze@dartmouth.edu',
      license='MIT',
      packages=['qspems'],
      install_requires=['numpy'],
      setup_requires=['wheel'],
      zip_safe=False)