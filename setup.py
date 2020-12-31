from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='hfta',
    version='0.1.0',
    description='Horizontally Fused Training Array',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/UofT-EcoSystem/hfta',
    author='Shang Wang,Peiming Yang,Yuxuan Zheng,Xin Li',
    author_email='wangsh46@cs.toronto.edu,'
    'yangpm1999@sjtu.edu.cn,'
    'yuxuan.zheng@mail.utoronto.ca,'
    'nix.li@mail.utoronto.ca',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'timing_parser=hfta.workflow.timing:timing_parser_main',
            'dcgm_parser=hfta.workflow.dcgm_monitor:dcgm_parser_main'
        ],
    },
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.1.5', 'numpy', 'scipy', 'matplotlib', 'psutil', 'torch>=1.6.0'
    ],
)
