from setuptools import setup

setup(
	name='package',
	version='0.1',
	description='Startup package',
	author='Vlad Myrhorodskyi',
	author_email='mirgorodskiy295@gmail.com',
	packages=['package.feature', 'package.ml_training'],
	install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'mlflow']
)